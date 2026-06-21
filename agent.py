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
except ImportError:
    logger.warning("livekit-plugins-google not installed")

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
                    end_of_speech_sensitivity=_gt.EndSensitivity.END_SENSITIVITY_HIGH,
                    silence_duration_ms=500,
                    prefix_padding_ms=100,
                ),
                activity_handling=_gt.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=_gt.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
            )
            _session_resumption_cfg = _gt.SessionResumptionConfig(transparent=True)
            _ctx_compression_cfg = _gt.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=_gt.SlidingWindow(target_tokens=12800),
            )
            # ── DISABLE THINKING — biggest single latency win ──────────────────
            # Gemini 2.5/3.x native-audio defaults to extended "thinking" before
            # each response (2-5s of internal reasoning). For a voice agent that's
            # unacceptable — kill it. thinking_budget=0 disables the think step.
            try:
                _thinking_cfg = _gt.ThinkingConfig(thinking_budget=0)
            except Exception:
                _thinking_cfg = None
            logger.info("Fast VAD applied (HIGH sens, 500ms silence) + thinking DISABLED")
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

    DIAL-FIRST PATTERN — CRITICAL:
    Start Gemini Live ONLY after create_sip_participant(wait_until_answered=True) completes.
    If you start the session during ring time (~20-30s), the Gemini idle timeout fires
    and the session dies silently before the call is even answered.

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

    # ── PRE-FETCH CONTACT HISTORY (runs BEFORE session.start) ─────────────────
    # We pull call history / appointments / memories from Supabase concurrently
    # and inject the result into the system prompt. The model no longer needs
    # to call lookup_contact mid-greeting — that was blocking the greeting for
    # 10-15s while the tool call resolved. By the time the LLM is asked to
    # speak, history is already in its context.
    contact_history_text = ""
    if phone_number:
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
            calls = calls if isinstance(calls, list) else []
            appointments = appointments if isinstance(appointments, list) else []
            memories = memories if isinstance(memories, list) else []
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
                contact_history_text = "\n".join(lines)
                await _log("info", f"Pre-fetched history for {phone_number}: {len(calls)} calls, {len(appointments)} appts, {len(memories)} memories")
            else:
                contact_history_text = "\n\n━━━ KNOWN CONTACT HISTORY ━━━\nNo prior history — first-time contact."
        except Exception as exc:
            logger.warning("Contact pre-fetch failed (non-fatal): %s", exc)

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

    # ── Connect ──────────────────────────────────────────────────────────────
    await ctx.connect()
    await _log("info", f"Connected to LiveKit room: {ctx.room.name}")

    # ── PARALLEL DIAL + AI STARTUP ──────────────────────────────────────────
    # Old flow:  dial(blocks 5-15s for ring) → session.start(1s) → greeting
    #            user perceived: ~6-16s of "AI not speaking" after pickup
    # New flow:  dial(fire-and-forget) → session.start(parallel with ring) →
    #            wait for participant_connected → greeting (instant)
    #            user perceived: ~1-2s after pickup — AI session is already warm

    trunk_id = os.getenv("OUTBOUND_TRUNK_ID") if phone_number else None
    if phone_number and not trunk_id:
        await _log("error", "OUTBOUND_TRUNK_ID not set — cannot place outbound call")
        ctx.shutdown()
        return

    # ── Build and start Gemini Live session FIRST (warm) ─────────────────────
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    await _log("info", f"Building AI session — model={gemini_model}")
    active_tools = tool_ctx.build_tool_list(enabled_tools)
    await _log("info", f"Tools loaded: {[t.__name__ for t in active_tools]}")
    session = _build_session(tools=active_tools, system_prompt=system_prompt)

    if _HAS_ROOM_OPTIONS:
        from livekit.agents import RoomOptions as _RO
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_options=_RO(input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony())),
        )
    else:
        _session_kwargs = dict(
            room=ctx.room,
            agent=OutboundAssistant(instructions=system_prompt),
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
        )

    # ── Events to coordinate dial + AI ───────────────────────────────────────
    _sip_identity = f"sip_{phone_number}" if phone_number else None
    _answered_event: asyncio.Event = asyncio.Event()
    _dial_failed_event: asyncio.Event = asyncio.Event()
    _dial_error: list = []

    if phone_number:
        def _on_participant_connected(p):
            if p.identity == _sip_identity:
                _answered_event.set()

        ctx.room.on("participant_connected", _on_participant_connected)

    # ── Kick off the SIP dial in the BACKGROUND (non-blocking) ────────────────
    async def _dial_bg():
        await _log("info", f"Dialing {phone_number} via SIP trunk {trunk_id} (non-blocking)")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=_sip_identity,
                    wait_until_answered=False,    # ← KEY: don't block; event fires on answer
                )
            )
        except Exception as exc:
            _dial_error.append(exc)
            _dial_failed_event.set()
            return
        # If the dial completed but the SIP participant never connected (busy/reject/no-answer),
        # the livekit server will fire a participant_disconnected for the SIP identity.
        # We rely on a 45s timeout to bail out if no participant_connected fires.
        try:
            await asyncio.wait_for(_answered_event.wait(), timeout=45)
            await _log("info", f"Call ANSWERED — {phone_number} picked up")
        except asyncio.TimeoutError:
            await _log("warning", f"No answer within 45s for {phone_number}")
            _dial_failed_event.set()

    dial_task = asyncio.create_task(_dial_bg())

    # ── Start Gemini session NOW (parallel with ring time) ───────────────────
    await session.start(**_session_kwargs)
    await _log("info", "Agent session started (parallel with ring)")

    # ── Optional S3 recording — BACKGROUND so it does NOT delay the greeting ──
    async def _start_recording_bg():
        if not phone_number:
            return
        _aws_key    = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID", "")
        _aws_secret = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        _aws_bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET_NAME", "")
        _s3_endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT", "")
        _s3_region  = os.getenv("S3_REGION") or os.getenv("AWS_REGION", "ap-northeast-1")
        if not (_aws_key and _aws_secret and _aws_bucket):
            return
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
            # ── STORE egress_id so we can stop it on call end (CRITICAL) ──────
            tool_ctx.recording_egress_id = _egress.egress_id
            await _log("info", f"Recording started (bg): egress={_egress.egress_id} -> {tool_ctx.recording_url}")
        except Exception as _exc:
            await _log("warning", f"Recording start failed (non-fatal): {_exc}")
    asyncio.create_task(_start_recording_bg())

    # ── Wait for the lead to answer OR dial to fail ──────────────────────────
    _t_answer = time.time()  # ← timestamp: lead picked up
    await asyncio.wait(
        {dial_task},
        timeout=50,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if _dial_failed_event.is_set() or (dial_task.done() and dial_task.exception()):
        err = _dial_error[0] if _dial_error else (dial_task.exception() if dial_task.done() else None)
        await _log("error", f"SIP dial FAILED for {phone_number}: {err}")
        try:
            await session.aclose()
        except Exception:
            pass
        ctx.shutdown()
        return
    _t_say_call = time.time()  # ← timestamp: about to inject greeting
    await _log("info", f"PERF pickup→say_call = {(_t_say_call - _t_answer)*1000:.0f}ms")

    # ── Greeting — use generate_reply() (say() requires a TTS model) ────────
    # session.say() needs a separate TTS model which we don't have in
    # native-audio Gemini Live mode. generate_reply() uses the LLM directly.
    if phone_number:
        _greeting_text = f"నమస్తే! నేను ప్రియ, {business_name} నుండి. మీరు {lead_name} గారా?"
    else:
        _greeting_text = "నమస్తే! నేను ప్రియ మాట్లాడుతున్నాను."
    try:
        _t_before_gr = time.time()
        await session.generate_reply(instructions=f"Say exactly this in Telugu: '{_greeting_text}'")
        _t_after_gr = time.time()
        await _log("info", f"PERF generate_reply returned in {(_t_after_gr - _t_before_gr)*1000:.0f}ms (pickup→say_call = {(_t_say_call - _t_answer)*1000:.0f}ms)")
    except Exception as _gr_exc:
        await _log("warning", f"generate_reply failed: {_gr_exc}")

    # ── Keep session alive until SIP participant actually leaves ─────────────
    if phone_number:
        _sip_identity = f"sip_{phone_number}"
        _disconnect_event = asyncio.Event()

        def _on_participant_disconnected(participant: rtc.RemoteParticipant):
            if participant.identity == _sip_identity:
                _disconnect_event.set()
        def _on_disconnected():
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
