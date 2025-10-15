from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.interview_manager import InterviewJobManager, InterviewJob
from app.models import InterviewJobConfig


def make_config(**overrides) -> InterviewJobConfig:
    base = {
        "project_name": "Test Project",
        "domain": "Wearables",
        "questions_per_persona": 5,
        "max_rounds": 5,
    }
    base.update(overrides)
    return InterviewJobConfig(**base)


class InterviewSheetMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdirs: list[tempfile.TemporaryDirectory] = []

    def tearDown(self) -> None:
        for tmp in self._tempdirs:
            tmp.cleanup()

    def _make_manager(self) -> InterviewJobManager:
        tmp = tempfile.TemporaryDirectory()
        self._tempdirs.append(tmp)
        return InterviewJobManager(Path(tmp.name))

    def test_session_id_grouping(self) -> None:
        manager = self._make_manager()
        cfg = make_config(tribe_count=12)
        job = InterviewJob(job_id="abcd1234ef56", config=cfg)

        session_ids = [manager._session_id_for_index(job, idx) for idx in range(cfg.tribe_count)]

        self.assertEqual(len(set(session_ids[:10])), 1)
        if cfg.tribe_count > 10:
            self.assertNotEqual(session_ids[0], session_ids[10])

    def test_build_tribe_sheet_rows_alignment(self) -> None:
        manager = self._make_manager()
        cfg = make_config(tribe_count=4)
        job = InterviewJob(job_id="feedbeef1234", config=cfg)
        headers = ["TribeName", "Gender", "SessionID"]
        categories = [
            {"name": "Active Explorers", "fields": {"TribeName": "Active Explorers", "Gender": "女性"}},
            {"name": "Budget Guardians", "fields": {"TribeName": "Budget Guardians", "Gender": "男性"}},
        ]

        tribes = manager._normalize_tribes(job, categories, headers)
        header_order, rows = manager._build_tribe_sheet_rows(tribes, headers)

        self.assertEqual(header_order[0], "TribeName")
        self.assertTrue(all(len(row) == len(header_order) for row in rows))
        tribe_name_idx = header_order.index("TribeName")
        session_idx = header_order.index("SessionID")
        self.assertEqual(rows[0][tribe_name_idx], "Active Explorers")
        self.assertEqual(rows[1][tribe_name_idx], "Budget Guardians")
        self.assertEqual(rows[0][session_idx], rows[1][session_idx])
        self.assertTrue(rows[0][session_idx].startswith(job.job_id[:8].upper()))

    def test_build_persona_sheet_rows_basic(self) -> None:
        manager = self._make_manager()
        cfg = make_config(tribe_count=1, persona_per_tribe=1)
        job = InterviewJob(job_id="facefeed5678", config=cfg)
        headers = ["TribeName", "Gender", "SessionID"]
        tribes = manager._normalize_tribes(job, [{"name": "Insight Seekers"}], headers)
        job.questions = ["What motivates you in this domain?"]

        personas = [
            {
                "persona_id": "persona_0001",
                "persona_sequence": 1,
                "tribe_id": 1,
                "persona_type": {
                    "age_band": "25-34",
                    "income_band": "mid",
                    "region": "JP_Kanto",
                    "attitude_cluster": "value_seeker",
                },
                "motivations": ["Stays active to improve wellbeing"],
                "frictions": ["Limited free time for research"],
                "tone": "conversational",
            }
        ]

        rows = manager._build_persona_sheet_rows(job, personas, tribes)

        self.assertTrue(rows)
        self.assertEqual(rows[0][0], "1_1")
        self.assertEqual(len(rows[0]), 2)
        self.assertIn("Tribe:", rows[0][1])
        self.assertIn("Shared Questions", rows[0][1])


if __name__ == "__main__":
    unittest.main()
