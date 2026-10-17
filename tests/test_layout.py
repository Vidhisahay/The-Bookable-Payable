from __future__ import annotations

import unittest

from ingestion.layout import layout_lines, layout_regions, normalized_words, resolve_party_roles, resolve_party_vat


def layout(*rows: tuple[str, float, float, float, float]) -> dict:
    words = []
    for line_no, (text, x0, y0, x1, y1) in enumerate(rows, 1):
        words.append({"text": text, "bbox": [x0, y0, x1, y1], "source": "ocr",
                      "confidence": 95.0, "block": 1, "line": line_no})
    return {"coordinate_system": "normalized_page", "words": words}


class SpatialLayoutTests(unittest.TestCase):
    def test_normalized_boxes_keep_source_confidence_and_page_coordinates(self):
        value = normalized_words([{"text": "Acme", "bbox": [10, 20, 30, 40], "confidence": 88.5}], 100, 100, source="native")
        self.assertEqual(value["words"][0]["bbox"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(value["words"][0]["source"], "native")
        self.assertEqual(value["words"][0]["confidence"], 88.5)

    def test_supplier_and_buyer_roles_follow_labels_when_columns_are_swapped(self):
        cases = [
            layout(("Bill To Northwind Services Ltd", 0.08, 0.2, 0.43, 0.23),
                   ("Supplier Silverbrook Media Ltd", 0.62, 0.5, 0.96, 0.53)),
            layout(("Supplier Silverbrook Media Ltd", 0.05, 0.7, 0.39, 0.73),
                   ("Bill To Northwind Services Ltd", 0.6, 0.15, 0.94, 0.18)),
        ]
        for item in cases:
            roles = resolve_party_roles(item)
            self.assertEqual(roles["supplier"]["status"], "found")
            self.assertEqual(roles["supplier"]["value"], "Silverbrook Media Ltd")
            self.assertEqual(roles["buyer"]["value"], "Northwind Services Ltd")

    def test_multiline_legal_name_is_joined_from_adjacent_same_column_lines(self):
        item = layout(("Supplier", 0.1, 0.2, 0.22, 0.23),
                      ("Asteria Consulting", 0.1, 0.235, 0.36, 0.26),
                      ("Limited", 0.1, 0.265, 0.2, 0.29))
        result = resolve_party_roles(item)["supplier"]
        self.assertEqual(result["status"], "found")
        self.assertIn("Asteria Consulting Limited", result["value"])

    def test_ship_to_and_payee_are_not_supplier_evidence(self):
        item = layout(("Ship To Harbor Works LLC", 0.1, 0.2, 0.4, 0.23),
                      ("Payee Meridian Bank Ltd", 0.6, 0.6, 0.92, 0.63))
        roles = resolve_party_roles(item)
        self.assertEqual(roles["supplier"]["status"], "missing")
        self.assertEqual(roles["ship_to"]["value"], "Harbor Works LLC")
        self.assertEqual(roles["payee"]["value"], "Meridian Bank Ltd")

    def test_conflicting_supplier_labels_remain_ambiguous(self):
        item = layout(("Supplier Asteria Consulting Ltd", 0.1, 0.2, 0.45, 0.23),
                      ("Supplier Beacon Trading LLC", 0.6, 0.7, 0.95, 0.73))
        result = resolve_party_roles(item)["supplier"]
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["value"])

    def test_competing_vat_ids_stay_with_their_spatial_party_blocks(self):
        item = layout(("Supplier Acme Services Ltd", .08, .2, .4, .23),
                      ("VAT Number", .08, .25, .23, .28),
                      ("GB123456789", .08, .29, .25, .32),
                      ("Bill To Northwind Trading LLC", .6, .2, .95, .23),
                      ("VAT Number", .6, .25, .76, .28),
                      ("GB987654321", .6, .29, .78, .32))
        self.assertEqual(resolve_party_roles(item)["supplier"]["value"], "Acme Services Ltd")
        supplier_vat = resolve_party_vat(item, "Acme Services Ltd")
        buyer_vat = resolve_party_vat(item, "Northwind Trading LLC")
        self.assertEqual((supplier_vat["status"], supplier_vat["value"]), ("found", "GB123456789"))
        self.assertEqual((buyer_vat["status"], buyer_vat["value"]), ("found", "GB987654321"))

    def test_two_equally_near_vat_ids_leave_supplier_vat_ambiguous(self):
        item = layout(("Supplier Acme Services Ltd", .1, .2, .4, .23),
                      ("VAT Number", .1, .25, .25, .28),
                      ("GB123456789", .1, .29, .3, .32),
                      ("VAT Number", .1, .25, .25, .28),
                      ("GB987654321", .1, .29, .3, .32))
        result = resolve_party_vat(item, "Acme Services Ltd")
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["value"])

    def test_competing_party_names_without_role_labels_are_unresolved(self):
        item = layout(("Asteria Consulting Ltd", 0.05, 0.2, 0.4, 0.23),
                      ("Beacon Trading LLC", 0.6, 0.7, 0.95, 0.73))
        result = resolve_party_roles(item)["supplier"]
        self.assertEqual(result["status"], "missing")

    def test_word_from_in_a_line_description_is_not_a_supplier_role_label(self):
        item = layout(("Supplier Oakhaven Supply Ltd", .25, .05, .5, .08),
                      ("Storage charges for goods from June", .1, .55, .8, .58))
        result = resolve_party_roles(item)["supplier"]
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["value"], "Oakhaven Supply Ltd")

    def test_unique_issuer_sharing_invoice_date_baseline_is_resolved_without_corner_rule(self):
        item = {"words": [
            {"text": word, "bbox": [x0, 0.2, x1, 0.23], "source": "ocr", "block": 1, "line": 1}
            for word, x0, x1 in [("Invoice", .25, .32), ("Date", .33, .38), ("Silverbrook", .55, .68),
                                 ("Media", .69, .75), ("Limited", .76, .84)]
        ] + [{"text": "Northwind Services UK Ltd", "bbox": [.1, .3, .4, .33],
              "source": "ocr", "block": 2, "line": 2}]}
        roles = resolve_party_roles(item)
        self.assertEqual(roles["supplier"]["status"], "found")
        self.assertEqual(roles["supplier"]["value"], "Silverbrook Media Limited")
        self.assertEqual(roles["supplier"]["evidence"]["label"], "Invoice Date adjacency")

    def test_explicit_bill_to_role_overrides_invoice_date_adjacency(self):
        item = {"words": [
            {"text": word, "bbox": [x0, .2, x1, .23], "source": "ocr", "block": 1, "line": 1}
            for word, x0, x1 in [("Invoice", .25, .32), ("Date", .33, .38), ("Acme", .55, .62), ("Limited", .63, .72)]
        ] + [{"text": "Bill To Acme Limited", "bbox": [.1, .28, .38, .31],
              "source": "ocr", "block": 2, "line": 2}]}
        roles = resolve_party_roles(item)
        self.assertEqual(roles["buyer"]["status"], "found")
        self.assertEqual(roles["supplier"]["status"], "missing")

    def test_layout_lines_reconstruct_words_by_visual_line_and_x_order(self):
        item = {"words": [
            {"text": "Limited", "bbox": [0.25, 0.1, 0.35, 0.12], "source": "native", "block": 1, "line": 1},
            {"text": "Asteria", "bbox": [0.1, 0.1, 0.24, 0.12], "source": "native", "block": 1, "line": 1},
        ]}
        self.assertEqual(layout_lines(item)[0]["text"], "Asteria Limited")

    def test_regions_keep_distinct_spatial_blocks_separate(self):
        item = {"words": [
            {"text": "Left", "bbox": [.1, .1, .2, .12], "source": "native", "block": 1, "line": 1},
            {"text": "Right", "bbox": [.7, .1, .8, .12], "source": "native", "block": 2, "line": 1},
        ]}
        regions = layout_regions(item)
        self.assertEqual([region["text"] for region in regions], ["Left", "Right"])


if __name__ == "__main__":
    unittest.main()
