from __future__ import annotations

from typing import Sequence

from .llm_intent_parser import IntentParse
from .semantic_gate import normalize_user_text


class HeuristicIntentBaseline:
    """Cheap baseline only; not the production parser.

    Its job is to make the benchmark executable without an API key and expose
    how much value a live LLM adds over simple rules.
    """

    META = (
        "uygula",
        "cevap verir misin",
        "sen cevapla",
        "ne yapayim",
        "farkli bir cevap",
        "onceki talimatlari yok say",
        "ignore",
        "parser",
    )

    QUESTION = ("ne olur", "ne olacak", "sence", "neden", "nasil", "mi ", "miyim")
    CLAIM_MARKERS = (
        "dedim",
        "soyledim",
        "soylemedim",
        "gordum",
        "dusunuyorum",
        "bence de o durumdaydi",
        "risk almayi sevmiyorum",
        "kontrol mekanizmasi olarak gordum",
        "bana soylemesi lazimdi",
        "hata ikimizde",
        "sucluyum",
        "sucsuzum",
    )

    def parse(
        self,
        raw_text: str,
        scene_context: str,
        allowed_actions: Sequence[str] = (),
    ) -> IntentParse:
        text = normalize_user_text(raw_text)
        context = normalize_user_text(scene_context)

        # Multi-slot scenes: several semicolon-like/current directives in one reply.
        if "uc karar" in context or "three decisions" in context:
            directive_hits = sum(
                token in text
                for token in ("ayirma", "kesinti olmasin", "kapat", "dokme", "devam", "gonder")
            )
            if directive_hits >= 2:
                return IntentParse("compound_action", None, "baseline", 0.7)

        if any(token in text for token in self.META):
            # In an active game scene, "uygula" refers to assistant meta-work, not in-world action.
            return IntentParse("meta", None, "baseline", 0.7)

        if "?" in raw_text or any(token in text for token in self.QUESTION):
            return IntentParse("question", None, "baseline", 0.6)

        if any(token in text for token in self.CLAIM_MARKERS):
            return IntentParse("claim", None, "baseline", 0.7)

        if "eger" in text and ("yapardim" in text or "olsa" in text):
            return IntentParse("hypothetical", None, "baseline", 0.6)

        # Context-sensitive current decision answers.
        if "ne yapardin" in context or "ne yapiyorsun" in context or "kararini ver" in context:
            if "controlled_run" in allowed_actions and ("kontrollu" in text or "calistirirdim" in text):
                return IntentParse("action", "controlled_run", "baseline", 0.8)
            if "stop_recheck" in allowed_actions and ("dokme" in text or "durdur" in text):
                return IntentParse("action", "stop_recheck", "baseline", 0.8)
            if "continue_monitor" in allowed_actions and ("kotulesirse" in text or "devam" in text):
                return IntentParse("action", "continue_monitor", "baseline", 0.8)
            if "send_home" in allowed_actions and ("gonder" in text or "cikar" in text):
                return IntentParse("action", "send_home", "baseline", 0.8)
            if "cancel_pour" in allowed_actions and ("iptal" in text or "program kaysin" in text):
                return IntentParse("action", "cancel_pour", "baseline", 0.8)
            if "inspect_logs" in allowed_actions and ("incele" in text and "log" in text):
                return IntentParse("action", "inspect_logs", "baseline", 0.8)
            if "continue_verification" in allowed_actions and "dogrulamaya devam" in text:
                return IntentParse("action", "continue_verification", "baseline", 0.8)
            if "call_person" in allowed_actions and ("arayin" in text or "arayin" in text):
                return IntentParse("action", "call_person", "baseline", 0.8)
            if "wait_patient" in allowed_actions and ("gitmeyin" in text or "bekleyin" in text):
                return IntentParse("action", "wait_patient", "baseline", 0.8)

        if "yapardim" in text or "olsaydi" in text or "olsa" in text:
            return IntentParse("hypothetical", None, "baseline", 0.5)

        if len(text) < 3 or text in {"hmm", "...", "bilmiyorum"}:
            return IntentParse("ambiguous", None, "baseline", 0.2)

        return IntentParse("ambiguous", None, "baseline", 0.2)
