#!/usr/bin/env python3
"""Regression checks for evidence-aware AI/IP case discovery."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import discover_cases  # noqa: E402


def recap_document(
    snippet: str,
    *,
    description: str = "Complaint",
    short_description: str = "",
    document_id: int = 1,
) -> dict[str, Any]:
    return {
        "id": document_id,
        "document_type": "PACER Document",
        "document_number": "1",
        "attachment_number": None,
        "description": description,
        "short_description": short_description,
        "snippet": snippet,
    }


def recap_result(
    docket_id: int,
    case_name: str,
    documents: list[dict[str, Any]],
    *,
    cause: str = "",
    suit_nature: str = "",
) -> dict[str, Any]:
    return {
        "docketNumber": f"1:26-cv-{docket_id % 100000:05d}",
        "docket_id": docket_id,
        "caseName": case_name,
        "court": "District Court, D. Test",
        "court_id": "testd",
        "dateFiled": "2026-07-29",
        "cause": cause,
        "suitNature": suit_nature,
        "recap_documents": documents,
        "meta": {"more_docs": False},
    }


def candidate_from(result: dict[str, Any], source: str = '"AI" "patent infringement"') -> dict[str, Any]:
    candidate = discover_cases.result_to_candidate(result, source)
    if candidate is None:
        raise AssertionError("valid RECAP fixture did not produce a discovery candidate")
    return candidate


class RecordingCourtListenerClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"results": [], "next": None}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((url, params))
        return self.response


class DiscoveryEvidenceTests(unittest.TestCase):
    def test_search_requests_recap_results_with_highlighting(self) -> None:
        client = RecordingCourtListenerClient()

        results, next_url = discover_cases.search_case_page(
            client,  # type: ignore[arg-type]
            '"machine learning" patent infringement',
            "2026-07-20",
            "2026-07-30",
        )

        self.assertEqual(results, [])
        self.assertEqual(next_url, "")
        self.assertEqual(len(client.calls), 1)
        _, params = client.calls[0]
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(params["type"], "r")
        self.assertEqual(params["highlight"], "on")

    def test_nested_recap_documents_are_aggregated_as_candidate_evidence(self) -> None:
        result = recap_result(
            73687886,
            "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
            [
                recap_document(
                    "COMPLAINT FOR PATENT INFRINGEMENT.",
                    description="Complaint",
                    document_id=10,
                ),
                recap_document(
                    "Hornetsecurity's 365 Total Protection enterprise plan includes "
                    "AI Recipient Validation and directly infringes the asserted patent.",
                    description="Complaint attachment",
                    document_id=11,
                ),
            ],
            cause="35:271 Patent Infringement",
        )

        candidate = candidate_from(result)

        self.assertIn("COMPLAINT FOR PATENT INFRINGEMENT", candidate["snippet"])
        self.assertIn("AI Recipient Validation", candidate["snippet"])

    def test_wyoming_machine_learning_patent_is_relevant(self) -> None:
        candidate = candidate_from(
            recap_result(
                73676038,
                "WYOMING TECHNOLOGY LICENSING, LLC v. BMW OF NORTH AMERICA, LLC",
                [
                    recap_document(
                        "The claim chart alleges that BMW's accused vehicle-control "
                        "apparatus uses a machine learning model and infringes the "
                        "asserted U.S. patent.",
                        description="Exhibit L - Patent Claim Chart",
                    )
                ],
                cause="35:271 Patent Infringement",
            ),
            source='"machine learning" patent infringement',
        )

        classification = discover_cases.fallback_classification(candidate)

        self.assertTrue(classification["relevant"])
        self.assertIn("patent infringement", classification["claims"])

    def test_gatekeeper_ai_recipient_validation_patent_is_relevant(self) -> None:
        candidate = candidate_from(
            recap_result(
                73687886,
                "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
                [
                    recap_document(
                        "COMPLAINT FOR PATENT INFRINGEMENT. Hornetsecurity manages "
                        "and provides a 365 Total Protection enterprise plan which "
                        "includes AI Recipient Validation. Hornetsecurity directly "
                        "infringes the system claims of the asserted patent."
                    )
                ],
                cause="35:271 Patent Infringement",
            )
        )

        classification = discover_cases.fallback_classification(candidate)

        self.assertTrue(classification["relevant"])
        self.assertIn("patent infringement", classification["claims"])

    def test_administrative_ai_standing_order_does_not_make_patent_case_relevant(self) -> None:
        candidate = candidate_from(
            recap_result(
                73670001,
                "Conventional Patent Owner LLC v. Device Company, Inc.",
                [
                    recap_document(
                        "Counsel must certify whether generative artificial "
                        "intelligence was used to prepare any court filing.",
                        description=(
                            "Standing Order Regarding the Use of Generative "
                            "Artificial Intelligence"
                        ),
                    )
                ],
                cause="35:271 Patent Infringement",
            ),
            source='"generative AI" patent',
        )

        classification = discover_cases.fallback_classification(candidate)

        self.assertFalse(classification["relevant"])

    def test_administrative_order_is_filtered_without_losing_substantive_ai_evidence(self) -> None:
        candidate = candidate_from(
            recap_result(
                73687886,
                "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
                [
                    recap_document(
                        "Counsel must disclose the use of generative AI in filings.",
                        description="Standing Order Regarding Generative AI",
                        document_id=20,
                    ),
                    recap_document(
                        "COMPLAINT FOR PATENT INFRINGEMENT. The accused email-security "
                        "product includes AI Recipient Validation and directly "
                        "infringes the asserted patent.",
                        description="Complaint",
                        document_id=21,
                    ),
                ],
                cause="35:271 Patent Infringement",
            )
        )

        classification = discover_cases.fallback_classification(candidate)

        self.assertTrue(classification["relevant"])
        self.assertIn("AI Recipient Validation", candidate["snippet"])

    def test_ai_product_liability_and_non_ip_cases_remain_irrelevant(self) -> None:
        fixtures = {
            "tiktok": recap_result(
                73678127,
                "Scholl v. TIKTOK INC.",
                [
                    recap_document(
                        "Plaintiff alleges that engagement algorithms and "
                        "industry-leading AI features harmed minor users."
                    )
                ],
                cause="28:1332 Diversity-Product Liability",
                suit_nature="Personal Injury - Product Liability",
            ),
            "novo": recap_result(
                73683342,
                "BRADLEY v. NOVO NORDISK A/S",
                [
                    recap_document(
                        "Novo allegedly used algorithms and machine learning to "
                        "optimize pharmaceutical marketing."
                    )
                ],
                cause="28:1332 Diversity-Product Liability",
                suit_nature="Personal Injury - Product Liability",
            ),
            "haley": recap_result(
                73674268,
                "Haley v. Matyjakowski",
                [
                    recap_document(
                        "Plaintiff alleges degradation of AI-assisted litigation "
                        "support tools and activity from identified IP addresses."
                    )
                ],
                cause="42:1983 Civil Rights",
                suit_nature="Civil Rights",
            ),
        }

        for label, result in fixtures.items():
            with self.subTest(case=label):
                classification = discover_cases.fallback_classification(
                    candidate_from(result, source='"artificial intelligence"')
                )
                self.assertFalse(classification["relevant"])
                self.assertEqual(classification["claims"], [])

    def test_duplicate_docket_hits_merge_distinct_evidence(self) -> None:
        first = recap_result(
            73687886,
            "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
            [
                recap_document(
                    "COMPLAINT FOR PATENT INFRINGEMENT.",
                    description="Complaint",
                    document_id=30,
                )
            ],
            cause="35:271 Patent Infringement",
        )
        second = recap_result(
            73687886,
            "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
            [
                recap_document(
                    "The accused product includes AI Recipient Validation and "
                    "directly infringes the asserted patent.",
                    description="Complaint",
                    document_id=31,
                )
            ],
            cause="35:271 Patent Infringement",
        )

        def search_page(
            _client: object,
            query: str,
            _after: str,
            _before: str,
            _page_url: str,
        ) -> tuple[list[dict[str, Any]], str]:
            return ([first], "") if query == "generic patent" else ([second], "")

        with patch.object(discover_cases, "search_case_page", side_effect=search_page):
            candidates, cursor, page_url, rate_limited, cap_reached = (
                discover_cases.collect_query_candidates(
                    object(),
                    "2026-07-20",
                    "2026-07-30",
                    set(),
                    5,
                    0,
                    queries=["generic patent", '"AI" "patent infringement"'],
                )
            )

        self.assertEqual(cursor, 2)
        self.assertEqual(page_url, "")
        self.assertFalse(rate_limited)
        self.assertFalse(cap_reached)
        self.assertEqual(list(candidates), ["id:73687886"])
        candidate = candidates["id:73687886"]
        self.assertIn("COMPLAINT FOR PATENT INFRINGEMENT", candidate["snippet"])
        self.assertIn("AI Recipient Validation", candidate["snippet"])
        self.assertTrue(discover_cases.fallback_classification(candidate)["relevant"])

    def test_pending_candidate_round_trip_preserves_aggregated_evidence(self) -> None:
        candidate = candidate_from(
            recap_result(
                73687886,
                "Gatekeeper Solutions, Inc. v. HornetSecurity, Inc.",
                [
                    recap_document(
                        "COMPLAINT FOR PATENT INFRINGEMENT.",
                        document_id=40,
                    ),
                    recap_document(
                        "The 365 Total Protection product includes AI Recipient "
                        "Validation and directly infringes the asserted patent.",
                        description="Complaint attachment",
                        document_id=41,
                    ),
                ],
                cause="35:271 Patent Infringement",
            )
        )

        pending, invalid = discover_cases.normalized_pending_candidate_state([candidate])

        self.assertFalse(invalid)
        self.assertEqual(len(pending), 1)
        self.assertIn("AI Recipient Validation", pending[0]["snippet"])
        self.assertTrue(discover_cases.fallback_classification(pending[0])["relevant"])


if __name__ == "__main__":
    unittest.main()
