import asyncio
import io
import json
import logging
import os
import ssl
import struct
import tempfile
import time
import wave
import certifi
from typing import Optional

from dotenv import load_dotenv

# Patch SSL before any network import
_orig_ssl = ssl.create_default_context
def _certifi_ssl(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    if not kwargs.get("cafile") and not kwargs.get("capath") and not kwargs.get("cadata"):
        kwargs["cafile"] = certifi.where()
    return _orig_ssl(purpose, **kwargs)
ssl.create_default_context = _certifi_ssl

from livekit import agents, api, rtc
from livekit.agents import Agent, AgentSession, RoomInputOptions
try:
    from livekit.agents import RoomOptions as _RoomOptions
    _HAS_ROOM_OPTIONS = True
except ImportError:
    _HAS_ROOM_OPTIONS = False
from livekit.plugins import noise_cancellation, silero

from db import init_db, log_error, log_call, get_enabled_tools
from prompts import build_prompt
from tools import AppointmentTools

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")

SIP_DOMAIN = os.getenv("VOBIZ_SIP_DOMAIN", "")


async def _log(level: str, msg: str, detail: str = "") -> None:
    if level == "info":      logger.info(msg)
    elif level == "warning": logger.warning(msg)
    else:                    logger.error(msg)
    try:
        await log_error("agent", msg, detail, level)
    except Exception:
        pass


def load_db_settings_to_env() -> None:
    """Load Supabase settings table into os.environ before worker starts."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        return
    try:
        from supabase import create_client
        client = create_client(url, key)
        result = client.table("settings").select("key, value").execute()
        for row in (result.data or []):
            if row.get("value"):
                os.environ[row["key"]] = row["value"]
    except Exception as exc:
        logger.warning("Could not load settings from Supabase: %s", exc)


# ── Import Google plugin paths ───────────────────────────────────────────────
_google_realtime = None
_google_beta_realtime = None
_google_llm = None
_google_tts = None

try:
    from livekit.plugins import google as _gp
    try:
        _google_realtime = _gp.realtime.RealtimeModel
        logger.info("Loaded google.realtime.RealtimeModel (stable path)")
    except AttributeError:
        pass
    try:
        _google_beta_realtime = _gp.beta.realtime.RealtimeModel
        logger.info("Loaded google.beta.realtime.RealtimeModel (beta path)")
    except AttributeError:
        pass
    try:
        _google_llm = _gp.LLM
        _google_tts = _gp.TTS
    except AttributeError:
        pass
except Exception as exc:
    logger.warning("livekit.plugins.google import failed: %s", exc)
    try:
        from livekit.plugins.google.realtime import RealtimeModel as _google_realtime
        logger.info("Loaded RealtimeModel via direct realtime import")
    except Exception as exc2:
        logger.error("Google realtime fallback import failed: %s", exc2)

_deepgram_stt = None
try:
    from livekit.plugins import deepgram as _dg
    _deepgram_stt = _dg.STT
except ImportError:
    pass


# ── Session factory ──────────────────────────────────────────────────────────

def _build_session(tools: list, system_prompt: str) -> AgentSession:
    """
    Build AgentSession with Gemini Live or pipeline fallback.

    CRITICAL SILENCE-PREVENTION CONFIG — all 3 required:
    1. SessionResumptionConfig(transparent=True) → auto-reconnects after timeout
    2. ContextWindowCompressionConfig → sliding window prevents token limit freeze
    3. RealtimeInputConfig(END_SENSITIVITY_LOW) → less aggressive VAD, 2s silence threshold

    ⚠️ EndSensitivity MUST use full string form: END_SENSITIVITY_LOW (not .LOW — AttributeError!)
    """
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    gemini_voice = os.getenv("GEMINI_TTS_VOICE", "Aoede")
    use_realtime = os.getenv("USE_GEMINI_REALTIME", "true").lower() != "false"

    RealtimeClass = _google_realtime or (_google_beta_realtime if use_realtime else None)

    if use_realtime and RealtimeClass is not None:
        logger.info("SESSION MODE: Gemini Live realtime (%s, voice=%s)", gemini_model, gemini_voice)
        try:
            from google.genai import types as _gt
            # ── LATENCY-OPTIMIZED VAD ──────────────────────────────────────────
            # HIGH sensitivity = detect end-of-speech FAST (responsiveness > robustness)
            # silence_duration_ms=500 = process speech after 0.5s pause (was 2000ms = 2s)
            # prefix_padding_ms=100 = minimal audio capture before speech (was 200ms)
            _realtime_input_cfg = _gt.RealtimeInputConfig(
                automatic_activity_detection=_gt.AutomaticActivityDetection(
                    end_of_speech_sensitivity=_gt.EndSensitivity.END_SENSITIVITY_LOW,
                    silence_duration_ms=800,
                    prefix_padding_ms=200,
                ),
                activity_handling=_gt.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=_gt.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
            )
            _session_resumption_cfg = _gt.SessionResumptionConfig(transparent=True)
            _ctx_compression_cfg = _gt.ContextWindowCompressionConfig(
                trigger_tokens=15000,
                sliding_window=_gt.SlidingWindow(target_tokens=8000),
            )
            # ── DISABLE THINKING — biggest single latency win ──────────────────
            # Gemini 2.5/3.x native-audio defaults to extended "thinking" before
            # each response (2-5s of internal reasoning). For a voice agent that's
            # unacceptable — kill it. thinking_budget=0 disables the think step.
            try:
                _thinking_cfg = _gt.ThinkingConfig(thinking_budget=0)
            except Exception:
                _thinking_cfg = None
            logger.info("VAD applied (LOW sens, 800ms silence, 200ms prefix) + thinking DISABLED + 15k token compression trigger")
        except Exception as _cfg_err:
            logger.warning("Could not build latency config: %s", _cfg_err)
            _realtime_input_cfg = None
            _session_resumption_cfg = None
            _ctx_compression_cfg = None
            _thinking_cfg = None

        realtime_kwargs: dict = dict(model=gemini_model, voice=gemini_voice, instructions=system_prompt)
        if _realtime_input_cfg is not None:
            realtime_kwargs["realtime_input_config"]      = _realtime_input_cfg
            realtime_kwargs["session_resumption"]         = _session_resumption_cfg
            realtime_kwargs["context_window_compression"] = _ctx_compression_cfg
        if _thinking_cfg is not None:
            realtime_kwargs["thinking_config"] = _thinking_cfg

        return AgentSession(llm=RealtimeClass(**realtime_kwargs), tools=tools)

    if _google_llm is None:
        raise RuntimeError("No Google AI backend. Run: pip install 'livekit-plugins-google>=1.0'")

    logger.info("SESSION MODE: pipeline (Deepgram STT + Gemini LLM + Google TTS)")
    stt = _deepgram_stt(model="nova-3", language="multi") if _deepgram_stt else None
    tts = _google_tts() if _google_tts else None
    return AgentSession(stt=stt, llm=_google_llm(model="gemini-2.0-flash"), tts=tts, vad=silero.VAD.load(), tools=tools)


class OutboundAssistant(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)


# ── CALL RECORDER ────────────────────────────────────────────────────────────
# Self-contained recorder that runs inside the agent process — bypasses
# LiveKit's egress service entirely (which has a monthly minutes quota
# that we kept hitting). Subscribes to every audio track in the room, writes
# raw PCM to a temp file, and on finalize() returns WAV bytes ready for
# Supabase Storage upload.
#
# Why not the LiveKit egress service?
#   • Quota: 49 prior egress attempts had used up the project's monthly
#     minutes. start_egress now returns HTTP 429.
#   • Reliability: the egress service runs on LiveKit's side. If their
#     infra hiccups, recordings vanish. Self-recording stays local until
#     we explicitly upload it.
#
# Output: 48kHz mono 16-bit PCM wrapped in a WAV header. ~5.7 MB/min.
# Format: WAV (instead of OGG) so the dashboard can <audio src> it directly.

class CallRecorder:
    def __init__(self, room, sample_rate: int = 48000, num_channels: int = 1):
        self.room = room
        # Fallbacks — _consume() detects the real values from the first frame
        # so the WAV header always matches the actual audio data.
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        # Per-track byte buffers so we can MIX (sum int16 samples with clipping)
        # all subscribed tracks at finalize time. Without this, writing both
        # AI + lead tracks to the same file made the WAV twice as long as
        # the call → playback at half speed.
        self._data_by_track: dict[str, bytearray] = {}
        self._subscribed: set = set()
        self._tasks: list = []
        self._stopped = False
        self._detected_sample_rate: Optional[int] = None
        self._detected_num_channels: Optional[int] = None
        # Stream raw PCM to per-track buffers (kept in memory for fast mixing)
        self._tmpfile = tempfile.NamedTemporaryFile(prefix="call_rec_", suffix=".pcm", delete=False)
        self._tmpfile_path = self._tmpfile.name
        self._tmpfile.close()

    async def start(self) -> None:
        """Subscribe to existing + future audio tracks in the room."""
        await self._scan_existing_tracks()
        self.room.on("track_subscribed", self._on_track_subscribed)

    async def _scan_existing_tracks(self) -> None:
        participants = [self.room.local_participant] + list(self.room.remote_participants.values())
        for p in participants:
            for pub in p.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    self._subscribe(pub.track)

    def _on_track_subscribed(self, track, publication, participant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self._subscribe(track)

    def _subscribe(self, track) -> None:
        if track.sid in self._subscribed:
            return
        self._subscribed.add(track.sid)
        self._data_by_track[track.sid] = bytearray()
        # Explicitly request 48kHz mono from AudioStream so we know exactly
        # what rate we'll write to the WAV. If the SDK can't resample, we
        # fall back to whatever the track's native rate is and detect it
        # in _consume() — then write the WAV header with that real rate.
        try:
            stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
        except TypeError:
            try:
                stream = rtc.AudioStream(track)
            except Exception:
                return
        task = asyncio.create_task(self._consume(stream, track.sid))
        self._tasks.append(task)

    async def _consume(self, stream, track_sid: str) -> None:
        try:
            async for event in stream:
                if self._stopped:
                    break
                if event.frame and event.frame.data:
                    # Capture the actual sample rate / channel count from
                    # the first frame we see.
                    if self._detected_sample_rate is None and event.frame.sample_rate:
                        self._detected_sample_rate = event.frame.sample_rate
                        self._detected_num_channels = event.frame.num_channels
                        logger.info(
                            f"CallRecorder detected: sample_rate={self._detected_sample_rate} "
                            f"num_channels={self._detected_num_channels} (track={track_sid[:8]})"
                        )
                    # Append to THIS track's buffer — we'll mix all tracks
                    # at finalize() time so the file has the call's true duration.
                    self._data_by_track[track_sid].extend(event.frame.data)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(f"CallRecorder: track {track_sid} ended: {exc}")

    async def finalize(self) -> Optional[bytes]:
        """
        Stop recording, mix all subscribed tracks by summing int16 samples
        (with clipping), and wrap the result in a WAV header.

        Returns None if no audio was captured. Cleans up the temp file.

        Why we mix: when the AI speaks, its audio goes out as a published
        track. When the lead speaks, their phone sends audio as another
        track. If we wrote both to one file sequentially, the WAV would
        be ~2x the actual call duration and the player would treat it
        as 2x slow audio. Mixing (summing int16 samples with clipping)
        produces a single mono file that matches the call's true duration.
        """
        self._stopped = True
        # Cancel all consumer tasks
        for t in self._tasks:
            t.cancel()
        # Brief settle
        await asyncio.sleep(0.2)
        # Best-effort close of the temp file (no longer used but keep cleanup)
        try:
            if os.path.exists(self._tmpfile_path):
                os.unlink(self._tmpfile_path)
        except Exception:
            pass

        # Collect each track's bytes
        track_datas = [bytes(d) for d in self._data_by_track.values() if d]
        if not track_datas:
            logger.warning("CallRecorder: no audio captured")
            return None

        # Mix: sum int16 samples across tracks, clip to int16 range
        if len(track_datas) == 1:
            mixed_pcm = track_datas[0]
            n_tracks = 1
        else:
            try:
                import numpy as np
                # Decode each track as int16, pad shorter ones with zeros
                # (shorter track = was active for less time)
                int16_tracks = []
                max_len = 0
                for d in track_datas:
                    n_samples = len(d) // 2
                    arr = np.frombuffer(d, dtype=np.int16, count=n_samples)
                    int16_tracks.append(arr)
                    if len(arr) > max_len:
                        max_len = len(arr)
                mixed = np.zeros(max_len, dtype=np.int32)  # int32 to avoid overflow during sum
                for arr in int16_tracks:
                    mixed[:len(arr)] += arr
                # Clip to int16 range
                np.clip(mixed, -32768, 32767, out=mixed)
                mixed_pcm = mixed.astype(np.int16).tobytes()
                n_tracks = len(int16_tracks)
            except ImportError:
                # Fallback: pure-Python int16 summing (slower but no numpy dep)
                mixed_pcm = self._mix_pure_python(track_datas)
                n_tracks = len(track_datas)

        # Use the actual sample rate / channel count from the first frame
        wav_rate   = self._detected_sample_rate or self.sample_rate
        wav_chans  = self._detected_num_channels or self.num_channels
        if self._detected_sample_rate and self._detected_sample_rate != 48000:
            logger.warning(
                f"CallRecorder: audio native rate is {wav_rate}Hz (not 48kHz) — "
                f"WAV written at native rate for correct playback speed"
            )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(wav_chans)
            w.setsampwidth(2)              # 16-bit PCM
            w.setframerate(wav_rate)
            w.writeframes(mixed_pcm)

        wav_bytes = buf.getvalue()
        duration_sec = len(mixed_pcm) / (wav_rate * wav_chans * 2)
        per_track = ", ".join(
            f"{sid[:8]}={len(d)//1024}KB" for sid, d in self._data_by_track.items() if d
        )
        logger.info(
            f"CallRecorder finalized: {len(wav_bytes)//1024}KB WAV @ {wav_rate}Hz "
            f"{wav_chans}ch ≈ {duration_sec:.1f}s ({n_tracks} tracks mixed: {per_track})"
        )
        return wav_bytes

    @staticmethod
    def _mix_pure_python(track_datas: list) -> bytes:
        """Fallback mixer when numpy is unavailable."""
        int16_tracks = []
        max_len = 0
        for d in track_datas:
            n = len(d) // 2
            samples = list(struct.unpack(f"<{n}h", d))
            int16_tracks.append(samples)
            if len(samples) > max_len:
                max_len = len(samples)
        mixed = [0] * max_len
        for track in int16_tracks:
            for i, s in enumerate(track):
                v = mixed[i] + s
                if v > 32767: v = 32767
                elif v < -32768: v = -32768
                mixed[i] = v
        return struct.pack(f"<{len(mixed)}h", *mixed)


async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Main entrypoint. Called per job. Reads metadata JSON from ctx.job.metadata.

    PARALLEL DIAL + SESSION PATTERN (latency fix):
    Dial SIP in the background with wait_until_answered=False, and start Gemini Live
    session in parallel with ring time. By the time the lead picks up, the AI session
    is fully warmed up — no 10-15s cold-start silence.

    The _answered_event (set by the participant_connected handler when the SIP leg
    establishes) is the AUTHORITATIVE sync point before any greeting fires.
    dial_task may complete as soon as the SIP INVITE is sent, NOT when the lead picks up.

    NO close_on_disconnect — SIP legs have brief audio dropouts that look like disconnects.
    Instead, watch participant_disconnected event for the specific SIP identity.
    """
    await _log("info", f"Job started — room: {ctx.room.name}")

    phone_number: Optional[str] = None
    lead_name = "there"
    business_name = "our company"
    service_type = "our service"
    custom_prompt: Optional[str] = None
    voice_override: Optional[str] = None
    model_override: Optional[str] = None
    tools_override: Optional[str] = None

    if ctx.job.metadata:
        try:
            data = json.loads(ctx.job.metadata)
            phone_number   = data.get("phone_number")
            lead_name      = data.get("lead_name", lead_name)
            business_name  = data.get("business_name", business_name)
            service_type   = data.get("service_type", service_type)
            custom_prompt  = data.get("system_prompt")
            voice_override = data.get("voice_override")
            model_override = data.get("model_override")
            tools_override = data.get("tools_override")
        except (json.JSONDecodeError, AttributeError):
            await _log("warning", "Invalid JSON in job metadata")

    await _log("info", f"Call job received — phone={phone_number} lead={lead_name} biz={business_name}")

    system_prompt = build_prompt(lead_name=lead_name, business_name=business_name,
                                  service_type=service_type, custom_prompt=custom_prompt)

    # ── PARALLEL SETUP: connect + prefetch run at the same time ──────────────
    # LATENCY FIX: previously the Supabase contact-history gather (3 queries,
    # ~4-6s) ran BEFORE ctx.connect(), which delayed session.start() and left
    # Gemini Live cold when the lead picked up — causing the variable 1-5s
    # greeting delay. Now connect + prefetch run concurrently, so the ring
    # time (3-10s) covers both, and generate_reply fires within 200-500ms of
    # pickup (was 1.3s-5.1s).
    async def _prefetch():
        _history = ""
        if not phone_number:
            return _history
        try:
            from db import get_calls_by_phone, get_appointments_by_phone, get_contact_memory
            calls, appointments, memories = await asyncio.gather(
                get_calls_by_phone(phone_number),
                get_appointments_by_phone(phone_number),
                get_contact_memory(phone_number),
                return_exceptions=True,
            )
            # Treat per-query failures as empty (defensive — single bad query
            # shouldn't poison the whole contact history)
            calls        = calls        if isinstance(calls,        list) else []
            appointments = appointments if isinstance(appointments, list) else []
            memories     = memories     if isinstance(memories,     list) else []
            if calls or appointments or memories:
                lines = ["\n\n━━━ KNOWN CONTACT HISTORY (already retrieved — do not call lookup_contact) ━━━"]
                if memories:
                    lines.append("REMEMBERED:")
                    for m in memories[:10]:
                        lines.append(f"  • {m['insight']}")
                if calls:
                    lines.append("PAST CALLS:")
                    for c in calls[:5]:
                        ts = (c.get("timestamp") or "")[:16]
                        lines.append(f"  • {ts} — {c.get('outcome','?')}: {c.get('reason','')}")
                if appointments:
                    lines.append("APPOINTMENTS:")
                    for a in appointments[:3]:
                        lines.append(f"  • {a.get('date')} {a.get('time')} — {a.get('service')} [{a.get('status')}]")
                _history = "\n".join(lines)
                await _log("info", f"Pre-fetched history for {phone_number}: {len(calls)} calls, {len(appointments)} appts, {len(memories)} memories")
            else:
                _history = "\n\n━━━ KNOWN CONTACT HISTORY ━━━\nNo prior history — first-time contact."
        except Exception as exc:
            logger.warning("Contact pre-fetch failed (non-fatal): %s", exc)
        return _history

    await ctx.connect()
    contact_history_text = await _prefetch()
    await _log("info", f"Connected to LiveKit room: {ctx.room.name}")

    system_prompt = system_prompt + contact_history_text
    tool_ctx = AppointmentTools(ctx, phone_number, lead_name)

    if voice_override:
        os.environ["GEMINI_TTS_VOICE"] = voice_override
    if model_override:
        os.environ["GEMINI_MODEL"] = model_override

    if tools_override:
        try:
            enabled_tools = json.loads(tools_override)
        except Exception:
            enabled_tools = await get_enabled_tools()
    else:
        enabled_tools = await get_enabled_tools()

    # ── PARALLEL DIAL + AI STARTUP (background, non-blocking) ───────────────
    # Dial kicks off the SIP INVITE and returns immediately. Gemini Live session
    # starts below in parallel with ring time. _answered_event is the authoritative
    # sync point — set by the participant_connected handler when the lead picks up.
    if phone_number:
        trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
        if not trunk_id:
            await _log("error", "OUTBOUND_TRUNK_ID not set — cannot place outbound call")
            ctx.shutdown()
            return

        _sip_identity = f"sip_{phone_number}"
        _answered_event: asyncio.Event = asyncio.Event()

        def _on_participant_connected(p):
            if p.identity == _sip_identity:
                _answered_event.set()
                logger.info("SIP participant connected — _answered_event set")

        ctx.room.on("participant_connected", _on_participant_connected)

        async def _dial_bg():
            try:
                await ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=trunk_id,
                        sip_call_to=phone_number,
                        participant_identity=_sip_identity,
                        wait_until_answered=False,    # ← fire-and-forget; event fires on answer
                    )
                )
                await _log("info", f"Dialing {phone_number} via SIP trunk {trunk_id} (non-blocking)")
            except Exception as exc:
                await _log("error", f"SIP dial FAILED for {phone_number}: {exc}")

        asyncio.create_task(_dial_bg())
    else:
        # No outbound dial — for local/test mode, set the answered event immediately
        _sip_identity = None
        _answered_event = asyncio.Event()
        _answered_event.set()

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    await _log("info", f"Building AI session — model={gemini_model}")
    active_tools = tool_ctx.build_tool_list(enabled_tools)
    await _log("info", f"Tools loaded: {[t.__name__ for t in active_tools]}")
    session = _build_session(tools=active_tools, system_prompt=system_prompt)

    _room_input = RoomInputOptions(
        # NOTE: noise_cancellation intentionally omitted.
        # BVCTelephony adds ~200-500ms first-frame audio processing latency on
        # top of Gemini Live cold-start. With a native-audio model doing its own
        # AEC/NS, telephony-grade NC is redundant and dominates pickup→greeting.
        close_on_disconnect=False,
    )
    if _HAS_ROOM_OPTIONS:
        from livekit.agents import RoomOptions as _RO
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_options=_RO(input_options=_room_input),
        )
    else:
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_input_options=_room_input,
        )

    await session.start(**_session_kwargs)
    await _log("info", "Agent session started — AI ready")

    # ── EXPLICIT WAIT: ensure lead has picked up before any greeting fires ──
    # session.start() resolved (Gemini Live warm) but dial_task may have already
    # completed — the SIP INVITE returns as soon as it's sent, NOT when the lead
    # picks up. _answered_event is set by the participant_connected handler when
    # the SIP leg fully establishes. This wait is the authoritative sync point:
    # by the time it returns, BOTH conditions hold:
    #   1. Gemini Live session is fully ready (session.start() resolved)
    #   2. Lead has actually picked up (_answered_event set)
    _t_answer = time.time()
    try:
        await asyncio.wait_for(_answered_event.wait(), timeout=45)
        await _log("info", f"Call ANSWERED — {phone_number} picked up")
    except asyncio.TimeoutError:
        await _log("warning", f"No answer within 45s for {phone_number} — ending session")
        try:
            await session.aclose()
        except Exception:
            pass
        ctx.shutdown()
        return
    _t_say_call = time.time()
    await _log("info", f"PERF pickup→say_call = {(_t_say_call - _t_answer)*1000:.0f}ms")

    # ── Self-recorder (bypasses LiveKit egress quota) ───────────────────────
    # We used to call start_room_composite_egress here, but that hits LiveKit
    # Cloud's monthly minutes quota (HTTP 429 once exhausted). Now we run a
    # CallRecorder in this agent process — subscribes to all audio tracks,
    # writes raw PCM to a temp file, and uploads to Supabase Storage at end.
    recorder: Optional[CallRecorder] = None
    if phone_number:
        try:
            recorder = CallRecorder(ctx.room)
            await recorder.start()
            await _log("info", "Self-recorder started (no LiveKit egress minutes consumed)")
        except Exception as _exc:
            await _log("warning", f"Self-recorder start failed (non-fatal): {_exc}")
            recorder = None

    # gemini-3.1/2.5 speak autonomously from system prompt — never call generate_reply
    _active_model = os.getenv("GEMINI_MODEL", "")
    if "3.1" in _active_model or "2.5" in _active_model:
        await _log("info", "Gemini native-audio: model will greet autonomously from system prompt")
    else:
        greeting = (
            f"The call just connected. Greet the lead and ask if you're speaking with {lead_name}."
            if phone_number else "Greet the caller warmly."
        )
        try:
            await session.generate_reply(instructions=greeting)
        except Exception as _gr_exc:
            await _log("warning", f"generate_reply failed: {_gr_exc}")

    # ── Keep session alive until SIP participant actually leaves ─────────────
    if phone_number:
        _sip_identity = f"sip_{phone_number}"
        _disconnect_event = asyncio.Event()

        async def _handle_disconnect_with_grace():
            # Brief audio dropouts on SIP legs are common. Don't kill the session
            # on the first participant_disconnected — wait 3s and re-check whether
            # the participant is really gone. If they came back, _disconnect_event
            # stays unset and the session continues.
            await asyncio.sleep(3)
            try:
                _still_present = _sip_identity in {
                    p.identity for p in ctx.room.remote_participants.values()
                }
            except Exception:
                _still_present = False
            if not _still_present:
                await _log("info", f"Disconnect grace period expired — {_sip_identity} truly gone, ending session")
                _disconnect_event.set()
            else:
                logger.info(f"{_sip_identity} reconnected during grace period — session continues")

        def _on_participant_disconnected(participant: rtc.RemoteParticipant):
            if participant.identity == _sip_identity:
                logger.info(f"{_sip_identity} disconnected — starting 3s grace period")
                asyncio.create_task(_handle_disconnect_with_grace())
        def _on_disconnected():
            # Room-level disconnect is terminal (LiveKit Cloud dropped us) — no grace period
                    _disconnect_event.set()

        ctx.room.on("participant_disconnected", _on_participant_disconnected)
        ctx.room.on("disconnected", _on_disconnected)

        try:
            await asyncio.wait_for(_disconnect_event.wait(), timeout=3600)
        except asyncio.TimeoutError:
            await _log("warning", "Call reached 1-hour safety timeout — shutting down")

        await _log("info", f"SIP participant disconnected — ending session for {phone_number}")

        # ── FINALIZE SELF-RECORDER, UPLOAD TO SUPABASE STORAGE ─────────────
        # We replaced LiveKit's egress service with an in-process recorder.
        # Now we mix the recorded audio into WAV, upload to Supabase Storage,
        # and set the recording_url that log_call will use below.
        if recorder is not None:
            try:
                _wav = await recorder.finalize()
                if _wav:
                    _sb_url   = os.getenv("SUPABASE_URL", "")
                    _sb_key   = os.getenv("SUPABASE_SERVICE_KEY", "")
                    _bucket   = os.getenv("S3_BUCKET", "outboundai")
                    if _sb_url and _sb_key and _bucket:
                        from supabase import create_client
                        _sb = create_client(_sb_url, _sb_key)
                        _path = f"recordings/{ctx.room.name}.wav"
                        _sb.storage.from_(_bucket).upload(
                            path=_path,
                            file=_wav,
                            file_options={"content-type": "audio/wav", "x-upsert": "true"},
                        )
                        # Public URL works if the bucket is publicly readable.
                        # If the bucket is private, swap to:
                        #   _sb.storage.from_(_bucket).create_signed_url(_path, 60*60*24*30)
                        tool_ctx.recording_url = _sb.storage.from_(_bucket).get_public_url(_path)
                        await _log("info", f"Recording uploaded → {tool_ctx.recording_url} ({len(_wav)//1024}KB)")
                    else:
                        await _log("warning", "Supabase creds missing — recording not uploaded")
                        tool_ctx.recording_url = None
                else:
                    await _log("warning", "Recorder finalized with no audio — skipping upload")
                    tool_ctx.recording_url = None
            except Exception as _rec_exc:
                await _log("warning", f"Recording finalize/upload failed (non-fatal): {_rec_exc}")
                tool_ctx.recording_url = None

        # ── LOG THE CALL (now that we have the recording URL) ──────────────
        # If the AI invoked end_call tool, the outcome/reason/duration are
        # already on tool_ctx. Otherwise fall back to "completed" with the
        # generic "call_ended_before_end_call_tool" reason.
        if not getattr(tool_ctx, "_call_logged", False):
            _outcome  = getattr(tool_ctx, "_call_outcome",  "completed")
            _reason   = getattr(tool_ctx, "_call_reason",   "call_ended_before_end_call_tool")
            _duration = getattr(tool_ctx, "_call_duration", int(time.time() - tool_ctx._call_start_time))
            try:
                await log_call(
                    phone_number=phone_number or "unknown",
                    lead_name=lead_name,
                    outcome=_outcome,
                    reason=_reason,
                    duration_seconds=_duration,
                    recording_url=getattr(tool_ctx, "recording_url", None),
                )
                tool_ctx._call_logged = True
                await _log("info", f"Call logged: {phone_number} outcome={_outcome} recording={'yes' if tool_ctx.recording_url else 'no'}")
            except Exception as _log_exc:
                await _log("warning", f"log_call failed: {_log_exc}")

        await session.aclose()
        # ── Close the room so LiveKit doesn't keep it warm ──────────────────
        try:
            ctx.shutdown()
        except Exception:
            pass
    else:
        _done = asyncio.Event()
        ctx.room.on("disconnected", lambda: _done.set())
        try:
            await asyncio.wait_for(_done.wait(), timeout=3600)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    init_db()
    load_db_settings_to_env()

    # ── CONFIGURATION GUARD — fail loudly if provider/model/realtime mode drift ──
    import logging as _lg
    _cfg_log = _lg.getLogger("outbound-agent.config")
    _expected_provider = "google"            # livekit.plugins.google.*
    _expected_model    = "gemini-3.1-flash-live-preview"
    _use_realtime      = os.getenv("USE_GEMINI_REALTIME", "true").lower() != "false"
    _actual_model      = os.getenv("GEMINI_MODEL", "")
    _voice             = os.getenv("GEMINI_TTS_VOICE", "Aoede")

    _cfg_log.info("=" * 60)
    _cfg_log.info("PROVIDER  : %s realtime", _expected_provider)
    _cfg_log.info("MODE      : %s", "REALTIME (native-audio)" if _use_realtime else "PIPELINE (STT+LLM+TTS)")
    _cfg_log.info("MODEL     : %s", _actual_model)
    _cfg_log.info("VOICE     : %s", _voice)
    _cfg_log.info("LIVEKIT   : %s", os.getenv("LIVEKIT_URL", "(unset)"))
    _cfg_log.info("SUPABASE  : %s", os.getenv("SUPABASE_URL", "(unset)")[:40])
    _cfg_log.info("=" * 60)

    assert _use_realtime, (
        "USE_GEMINI_REALTIME must be 'true' — pipeline fallback requires DEEPGRAM_API_KEY "
        "and would degrade to Deepgram STT + Gemini LLM + Google TTS, which is not what we want."
    )
    assert _actual_model == _expected_model, (
        f"GEMINI_MODEL must be exactly '{_expected_model}' (got '{_actual_model}'). "
        f"Other Gemini Live models have different latency/quality profiles."
    )

    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="outbound-caller")
    )
