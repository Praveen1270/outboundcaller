import asyncio
import logging
import os
import time
from typing import Optional

from livekit import agents, api
from livekit.agents import llm

from db import (
    check_slot, get_next_available, insert_appointment, log_call, log_error,
    get_calls_by_phone, get_appointments_by_phone,
    add_contact_memory, get_contact_memory, compress_contact_memory,
)

logger = logging.getLogger("appointment-tools")

# Fallback string returned to the model when a tool times out (10s).
# Don't tell the user there's an error — just nudge the agent to keep talking.
_TOOL_TIMEOUT_MSG = "Tool request timed out. Please continue the conversation."


async def _log(msg: str, detail: str = "", level: str = "info") -> None:
    try:
        await log_error("agent", msg, detail, level)
    except Exception:
        pass


class AppointmentTools(llm.ToolContext):
    """All function tools available to the appointment-booking agent."""

    def __init__(self, ctx: agents.JobContext, phone_number: Optional[str] = None, lead_name: Optional[str] = None):
        self.ctx = ctx
        self.phone_number = phone_number
        self.lead_name = lead_name
        self._call_start_time = time.time()
        self._sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
        self.recording_url: Optional[str] = None
        super().__init__(tools=[])

    def build_tool_list(self, enabled: list) -> list:
        """Return tool methods filtered by the enabled list. Empty list = all enabled."""
        all_methods = [
            self.check_availability, self.book_appointment, self.end_call,
            self.transfer_to_human, self.send_sms_confirmation, self.lookup_contact,
            self.remember_details, self.book_calcom, self.cancel_calcom,
        ]
        if not enabled:
            return all_methods
        name_map = {m.__name__: m for m in all_methods}
        return [name_map[n] for n in enabled if n in name_map]

    # ── check_availability ────────────────────────────────────────────────────
    @llm.function_tool
    async def check_availability(self, date: str, time: str) -> str:
        """
        Check whether a date/time slot is available for booking.
        Call this BEFORE attempting to book whenever the lead proposes a date/time.
        date format: YYYY-MM-DD  |  time format: HH:MM (24-hour)
        Returns 'available' or 'unavailable: next available slot is <slot>'.
        """
        try:
            return await asyncio.wait_for(
                self._check_availability_actual(date, time), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("check_availability timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "Unable to check availability right now — please suggest a date and I will confirm."

    async def _check_availability_actual(self, date: str, time: str) -> str:
        if await check_slot(date, time):
            return "available"
        next_slot = await get_next_available(date, time)
        return f"unavailable: next available slot is {next_slot}"

    # ── book_appointment ──────────────────────────────────────────────────────
    @llm.function_tool
    async def book_appointment(self, name: str, phone: str, date: str, time: str, service: str) -> str:
        """
        Book an appointment after the lead has verbally confirmed date, time, and service.
        Call ONLY after the lead confirms all details.
        name: lead's full name | phone: with country code | date: YYYY-MM-DD | time: HH:MM | service: type
        """
        try:
            return await asyncio.wait_for(
                self._book_appointment_actual(name, phone, date, time, service), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("book_appointment timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "Technical issue saving the booking. Our team will confirm shortly."

    async def _book_appointment_actual(self, name: str, phone: str, date: str, time: str, service: str) -> str:
        booking_id = await insert_appointment(name, phone, date, time, service)
        if booking_id is None:
            return (
                f"Sorry, {date} at {time} was just booked by someone else. "
                f"Please call check_availability to find the next open slot."
            )
        return f"Confirmed! Booking ID: {booking_id}. See you on {date} at {time} for {service}."

    # ── end_call ──────────────────────────────────────────────────────────────
    @llm.function_tool
    async def end_call(self, outcome: str, reason: str = "") -> str:
        """
        End the call and log the outcome. ALWAYS call this before the call ends.
        outcome: 'booked' | 'not_interested' | 'wrong_number' | 'voicemail' | 'no_answer' | 'callback_requested'
        reason: brief description
        """
        try:
            return await asyncio.wait_for(
                self._end_call_actual(outcome, reason), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("end_call timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception as exc:
            logger.error("end_call failed: %s", exc)
            return "Call ending — log may not have written."

    async def _end_call_actual(self, outcome: str, reason: str) -> str:
        duration = int(time.time() - self._call_start_time)
        try:
            await log_call(
                phone_number=self.phone_number or "unknown",
                lead_name=self.lead_name, outcome=outcome, reason=reason,
                duration_seconds=duration, recording_url=self.recording_url,
            )
            self._call_logged = True
        except Exception as exc:
            logger.error("Failed to log call: %s", exc)
        try:
            await self.ctx.room.disconnect()
        except Exception:
            pass
        return "Call ended."

    # ── transfer_to_human ─────────────────────────────────────────────────────
    @llm.function_tool
    async def transfer_to_human(self, reason: str) -> str:
        """
        Transfer the call to a human agent via SIP REFER.
        Call when lead requests a human, is angry, or has a complex issue.
        reason: why you're transferring
        """
        try:
            return await asyncio.wait_for(
                self._transfer_to_human_actual(reason), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("transfer_to_human timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "Transfer failed. Please call us back directly."

    async def _transfer_to_human_actual(self, reason: str) -> str:
        destination = os.getenv("DEFAULT_TRANSFER_NUMBER", "")
        if not destination:
            return "Transfer unavailable: no fallback number configured."
        if "@" not in destination:
            clean = destination.replace("tel:", "").replace("sip:", "")
            destination = f"sip:{clean}@{self._sip_domain}" if self._sip_domain else f"tel:{clean}"
        elif not destination.startswith("sip:"):
            destination = f"sip:{destination}"
        participant_identity = f"sip_{self.phone_number}" if self.phone_number else None
        if not participant_identity:
            for p in self.ctx.room.remote_participants.values():
                participant_identity = p.identity
                break
        if not participant_identity:
            return "Transfer failed: could not identify caller."
        await self.ctx.api.sip.transfer_sip_participant(
            api.TransferSIPParticipantRequest(
                room_name=self.ctx.room.name,
                participant_identity=participant_identity,
                transfer_to=destination, play_dialtone=False,
            )
        )
        return "Transferring you to a human agent now. Please hold."

    # ── send_sms_confirmation ─────────────────────────────────────────────────
    @llm.function_tool
    async def send_sms_confirmation(self, phone: str, message: str) -> str:
        """
        Send SMS confirmation after a successful booking. Skips silently if Twilio not configured.
        phone: lead's phone | message: text to send
        """
        try:
            return await asyncio.wait_for(
                self._send_sms_confirmation_actual(phone, message), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("send_sms_confirmation timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "SMS delivery failed, but booking is confirmed."

    async def _send_sms_confirmation_actual(self, phone: str, message: str) -> str:
        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        from_num = os.getenv("TWILIO_FROM_NUMBER", "")
        if not (sid and token and from_num):
            return "SMS skipped: Twilio not configured."
        from twilio.rest import Client
        loop = asyncio.get_event_loop()
        client = Client(sid, token)
        await loop.run_in_executor(None, lambda: client.messages.create(body=message, from_=from_num, to=phone))
        return f"SMS sent to {phone}."

    # ── lookup_contact ────────────────────────────────────────────────────────
    @llm.function_tool
    async def lookup_contact(self, phone: str) -> str:
        """
        Look up a contact's full history. Call at the START of every call before engaging.
        phone: the lead's phone number with country code
        Returns call history, appointments, and remembered details.
        """
        try:
            return await asyncio.wait_for(
                self._lookup_contact_actual(phone), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("lookup_contact timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "Unable to retrieve contact history."

    async def _lookup_contact_actual(self, phone: str) -> str:
        # Run all three queries concurrently — saves ~200-400ms vs serial
        calls, appointments, memories = await asyncio.gather(
            get_calls_by_phone(phone),
            get_appointments_by_phone(phone),
            get_contact_memory(phone),
            return_exceptions=True,
        )
        # Defensive: if any one query raised, treat it as empty rather than blowing up
        if isinstance(calls, Exception):
            logger.warning("lookup_contact: calls fetch failed: %s", calls)
            calls = []
        if isinstance(appointments, Exception):
            logger.warning("lookup_contact: appointments fetch failed: %s", appointments)
            appointments = []
        if isinstance(memories, Exception):
            logger.warning("lookup_contact: memories fetch failed: %s", memories)
            memories = []
        if not calls and not appointments and not memories:
            return f"No history for {phone}. First-time contact."
        lines = [f"Contact history for {phone}:"]
        if memories:
            lines.append(f"\nREMEMBERED ({len(memories)} notes):")
            for m in memories[:10]:
                lines.append(f"  • {m['insight']}")
        if calls:
            lines.append(f"\nCALL HISTORY ({len(calls)} calls):")
            for c in calls[:5]:
                ts = (c.get("timestamp") or "")[:16]
                lines.append(f"  • {ts} — {c.get('outcome','?')}: {c.get('reason','')}")
        if appointments:
            lines.append(f"\nAPPOINTMENTS ({len(appointments)}):")
            for a in appointments[:3]:
                lines.append(f"  • {a.get('date')} {a.get('time')} — {a.get('service')} [{a.get('status')}]")
        return "\n".join(lines)

    # ── remember_details ──────────────────────────────────────────────────────
    @llm.function_tool
    async def remember_details(self, insight: str) -> str:
        """
        Store a key insight about this lead for future calls.
        Use whenever you learn something useful: preferences, objections, timing, family info.
        Examples: "Prefers morning calls", "Has 2 kids, interested in family plan", "Callback in 2 weeks"
        insight: the detail to remember
        """
        try:
            return await asyncio.wait_for(
                self._remember_details_actual(insight), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("remember_details timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception:
            return "Could not save detail."

    async def _remember_details_actual(self, insight: str) -> str:
        if not self.phone_number:
            return "Cannot remember — no phone number for this call."
        await add_contact_memory(self.phone_number, insight)
        memories = await get_contact_memory(self.phone_number)
        if len(memories) >= 5:
            asyncio.create_task(self._compress_memories())
        return f"Remembered: {insight}"

    # ── _compress_memories (internal helper, NOT a tool) ──────────────────────
    async def _compress_memories(self) -> None:
        try:
            memories = await get_contact_memory(self.phone_number)
            if len(memories) < 5:
                return
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                return
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            bullet_list = "\n".join(f"- {m['insight']}" for m in memories)
            prompt = f"Compress these notes about a sales contact into 3-5 concise bullets. Keep all key facts.\n\n{bullet_list}"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
            if response.text.strip():
                await compress_contact_memory(self.phone_number, response.text.strip())
        except Exception as exc:
            logger.warning("Memory compression failed: %s", exc)

    # ── book_calcom ───────────────────────────────────────────────────────────
    @llm.function_tool
    async def book_calcom(self, name: str, email: str, date: str, start_time: str, notes: str = "") -> str:
        """
        Book in Cal.com calendar after book_appointment succeeds.
        name: full name | email: lead's email | date: YYYY-MM-DD | start_time: HH:MM | notes: optional
        """
        try:
            return await asyncio.wait_for(
                self._book_calcom_actual(name, email, date, start_time, notes), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("book_calcom timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception as exc:
            return f"Cal.com booking failed: {exc}"

    async def _book_calcom_actual(self, name: str, email: str, date: str, start_time: str, notes: str) -> str:
        api_key = os.getenv("CALCOM_API_KEY", "")
        event_type_id = os.getenv("CALCOM_EVENT_TYPE_ID", "")
        timezone = os.getenv("CALCOM_TIMEZONE", "Asia/Kolkata")
        if not api_key or not event_type_id:
            return "Cal.com not configured — skipping. Add CALCOM_API_KEY and CALCOM_EVENT_TYPE_ID."
        from datetime import datetime as _dt
        start_dt = _dt.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.cal.com/v1/bookings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"eventTypeId": int(event_type_id), "start": start_iso, "timeZone": timezone,
                      "responses": {"name": name, "email": email, "notes": notes},
                      "metadata": {"source": "OutboundAI"}, "language": "en"},
            )
        data = resp.json()
        if resp.status_code not in (200, 201):
            raise ValueError(data.get("message") or str(data))
        uid = data.get("uid", "")
        return f"Cal.com booked. UID: {uid}"

    # ── cancel_calcom ─────────────────────────────────────────────────────────
    @llm.function_tool
    async def cancel_calcom(self, booking_uid: str, reason: str = "") -> str:
        """
        Cancel a Cal.com booking by UID.
        booking_uid: from book_calcom | reason: optional
        """
        try:
            return await asyncio.wait_for(
                self._cancel_calcom_actual(booking_uid, reason), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("cancel_calcom timed out after 10s")
            return _TOOL_TIMEOUT_MSG
        except Exception as exc:
            return f"Cancellation failed: {exc}"

    async def _cancel_calcom_actual(self, booking_uid: str, reason: str) -> str:
        api_key = os.getenv("CALCOM_API_KEY", "")
        if not api_key:
            return "Cal.com not configured."
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(
                f"https://api.cal.com/v1/bookings/{booking_uid}",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"reason": reason} if reason else {},
            )
        if resp.status_code not in (200, 204):
            raise ValueError(f"HTTP {resp.status_code}")
        return f"Cancelled Cal.com booking {booking_uid}."