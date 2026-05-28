from expert._compat import patch_collections

patch_collections()

from experta import KnowledgeEngine, MATCH, NOT, Rule, TEST

from expert.facts import Incident


class IncidentEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.action: str | None = None
        self.message: str = ""
        self.plan_source: str | None = None
        self.station_filter: str | None = None
        self.retrieval_query: str | None = None
        self.info_topics: list[str] = []
        self.pending_slot: str | None = None
        self.derived_actions: list[str] = []
        self.role_focus: str | None = None
        self.time_focus: str | None = None

    @Rule(
        Incident(event_type=MATCH.et),
        TEST(lambda et: et not in ("line_blockage", "station_disruption")),
        salience=100,
    )
    def unknown_event_type(self, et):
        self._ask(
            "Is this a **line blockage** between two stations, or a **station disruption** at one location?"
        )

    @Rule(
        NOT(Incident(event_type=MATCH.et)),
        NOT(Incident(from_station=MATCH.fs)),
        NOT(Incident(station=MATCH.st)),
        salience=99,
    )
    def need_event_type(self):
        self._ask(
            "I can help with SWR contingency plans. Is this a **line blockage** between two stations, "
            "or a **station disruption** at one station? And where is it?",
            pending_slot="event_type",
        )

    @Rule(
        NOT(Incident(event_type=MATCH.et)),
        Incident(from_station=MATCH.fs),
        NOT(Incident(to_station=MATCH.ts)),
        salience=91,
    )
    def need_type_after_partial_from(self, fs):
        self._ask(
            f"You mentioned **{fs}**. Is this a **line blockage** from {fs} to another station "
            f"(if so, which station?), or a **station disruption** at {fs}?",
            pending_slot="event_type",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        NOT(Incident(from_station=MATCH.fs)),
        salience=90,
    )
    def need_from_station(self):
        self._ask(
            "Which station is at the **start** of the blocked section?",
            pending_slot="from_station",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(from_station=MATCH.fs),
        NOT(Incident(to_station=MATCH.ts)),
        salience=89,
    )
    def need_to_station(self, fs):
        self._ask(
            f"Which station is at the **end** of the blocked section (after {fs})?",
            pending_slot="to_station",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(from_station=MATCH.fs),
        Incident(to_station=MATCH.ts),
        NOT(Incident(severity=MATCH.sev)),
        salience=88,
    )
    def need_severity_line(self, fs, ts):
        self._ask(
            f"For the blockage between **{fs}** and **{ts}**, is it a **full** blockage "
            "(both lines) or **partial** (one line)?",
            pending_slot="severity",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(from_station=MATCH.fs),
        Incident(to_station=MATCH.ts),
        Incident(severity=MATCH.sev),
        NOT(Incident(staff_role=MATCH.role)),
        salience=87,
    )
    def need_staff_role(self):
        self._ask(
            "Which role are you acting as: **signaller**, **station staff**, or **control**?",
            pending_slot="staff_role",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(severity=MATCH.sev),
        TEST(lambda sev: sev == "both_lines_blocked"),
        NOT(Incident(incident_time=MATCH.it)),
        salience=86,
    )
    def need_incident_time(self):
        self._ask(
            "What time is the incident (e.g. **08:30**, **6pm**, **morning**)?",
            pending_slot="incident_time",
        )

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(severity=MATCH.sev),
        Incident(incident_time=MATCH.it),
        NOT(Incident(duration_minutes=MATCH.dm)),
        salience=85,
    )
    def need_duration(self):
        self._ask(
            "How long has this lasted or is expected to last (e.g. **30 min**, **1 hour**)?",
            pending_slot="duration_minutes",
        )

    @Rule(
        Incident(event_type="station_disruption"),
        NOT(Incident(station=MATCH.st)),
        salience=90,
    )
    def need_station(self):
        self._ask("Which **station** is affected?", pending_slot="station")

    @Rule(
        Incident(event_type="line_blockage"),
        Incident(from_station=MATCH.fs),
        Incident(to_station=MATCH.ts),
        Incident(severity=MATCH.sev),
        salience=10,
    )
    def ready_line_blockage(self, fs, ts, sev):
        topics = self.info_topics or ["staff", "passengers"]
        self.role_focus = self.role_focus or self._best_role_focus()
        self.time_focus = self.time_focus or self._best_time_focus()
        actions = self._derive_actions(sev=sev)
        self.derived_actions = actions
        actions_text = ", ".join(actions) if actions else "none"
        self._set_retrieve(
            plan_source="cpt_presentation",
            station_filter=None,
            query=(
                f"Contingency plan line blockage between {fs} and {ts}. "
                f"Severity: {sev}. "
                f"Role focus: {self.role_focus or 'general'}. "
                f"Time focus: {self.time_focus or 'unspecified'}. "
                f"Derived actions: {actions_text}. "
                f"Information needed: {', '.join(topics)}. "
                f"Service alteration signaller station staff passenger."
            ),
            topics=topics,
        )

    @Rule(
        Incident(event_type="station_disruption"),
        Incident(station=MATCH.st),
        salience=10,
    )
    def ready_station_disruption(self, st):
        topics = self.info_topics or ["staff", "passengers"]
        self.role_focus = self.role_focus or self._best_role_focus()
        self.time_focus = self.time_focus or self._best_time_focus()
        self._set_retrieve(
            plan_source="station_plan",
            station_filter=st,
            query=(
                f"Active station disruption at {st} — immediate operational actions. "
                f"Role focus: {self.role_focus or 'general'}. "
                f"Time focus: {self.time_focus or 'unspecified'}. "
                f"Communication with control practical operation of the station "
                f"crowd welfare alternative transport buses taxis evacuation. "
                f"Staff and passenger steps during disruption."
            ),
            topics=topics,
        )

    def _ask(self, message: str, *, pending_slot: str | None = None) -> None:
        if self.action is not None:
            return
        self.action = "ask"
        self.message = message
        self.pending_slot = pending_slot

    def _set_retrieve(
        self,
        *,
        plan_source: str,
        station_filter: str | None,
        query: str,
        topics: list[str],
    ) -> None:
        if self.action == "retrieve":
            return
        self.action = "retrieve"
        self.plan_source = plan_source
        self.station_filter = station_filter
        self.retrieval_query = query
        self.info_topics = topics

    def _best_role_focus(self) -> str | None:
        for fact in self.facts.values():
            role = fact.get("staff_role")
            if role:
                return role
        return None

    def _best_time_focus(self) -> str | None:
        for fact in self.facts.values():
            service_period = fact.get("service_period")
            if service_period:
                return service_period
        for fact in self.facts.values():
            incident_time = fact.get("incident_time")
            if incident_time:
                return incident_time
        return None

    def _derive_actions(self, *, sev: str) -> list[str]:
        actions: list[str] = []
        if sev == "both_lines_blocked" and self._best_time_focus() == "peak":
            actions.extend(["bus_replacement", "welfare_focus"])

        duration = None
        for fact in self.facts.values():
            value = fact.get("duration_minutes")
            if value is not None:
                duration = value
                break
        if duration is not None and duration > 30:
            actions.append("welfare_escalation")
        if duration is not None and duration > 60:
            actions.append("refreshments_escalation")
        return actions
