import asyncio
import json
import logging
import os
import ssl
import time
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

    # ── Optional S3 recording ────────────────────────────────────────────────
    if phone_number:
        _aws_key    = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID", "")
        _aws_secret = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        _aws_bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET_NAME", "")
        _s3_endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT", "")
        _s3_region  = os.getenv("S3_REGION") or os.getenv("AWS_REGION", "ap-northeast-1")
        if _aws_key and _aws_secret and _aws_bucket:
            try:
                _recording_path = f"recordings/{ctx.room.name}.ogg"
                _egress_req = api.RoomCompositeEgressRequest(
                    room_name=ctx.room.name, audio_only=True,
                    file_outputs=[api.EncodedFileOutput(
                        file_type=api.EncodedFileType.OGG, filepath=_recording_path,
                        s3=api.S3Upload(access_key=_aws_key, secret=_aws_secret,
                                        bucket=_aws_bucket, region=_s3_region, endpoint=_s3_endpoint),
                    )],
                )
                _egress = await ctx.api.egress.start_room_composite_egress(_egress_req)
                _s3_ep = _s3_endpoint.rstrip("/")
                tool_ctx.recording_url = (f"{_s3_ep}/{_aws_bucket}/{_recording_path}"
                                           if _s3_ep else f"s3://{_aws_bucket}/{_recording_path}")
                tool_ctx.recording_egress_id = _egress.egress_id
                await _log("info", f"Recording started: egress={_egress.egress_id}")
            except Exception as _exc:
                await _log("warning", f"Recording start failed (non-fatal): {_exc}")

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
        # Fallback: ensure the call is logged even if the AI never invoked end_call tool
        if not getattr(tool_ctx, "_call_logged", False):
            try:
                _duration = int(time.time() - tool_ctx._call_start_time)
                await log_call(
                    phone_number=phone_number or "unknown",
                    lead_name=lead_name,
                    outcome="completed",
                    reason="call_ended_before_end_call_tool",
                    duration_seconds=_duration,
                    recording_url=getattr(tool_ctx, "recording_url", None),
                )
                tool_ctx._call_logged = True
                await _log("info", f"Fallback log_call() written for {phone_number} ({_duration}s)")
            except Exception as _fallback_exc:
                await _log("warning", f"Fallback log_call failed: {_fallback_exc}")

        # ── STOP EGRESS — this triggers the actual S3 upload ────────────────
        # Without this, LiveKit keeps the recording open indefinitely and the
        # file never lands in Supabase Storage.
        _egress_id = getattr(tool_ctx, "recording_egress_id", None)
        if _egress_id:
            try:
                # LiveKit API: stop_egress takes a StopEgressRequest, NOT kwarg
                await ctx.api.egress.stop_egress(api.StopEgressRequest(egress_id=_egress_id))
                await _log("info", f"Egress {_egress_id} stopped — uploading recording to S3")
            except Exception as _stop_exc:
                await _log("warning", f"Egress stop failed (recording may not upload): {_stop_exc}")

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
