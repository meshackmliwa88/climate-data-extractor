from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_removed_sidebar_features_are_absent():
    base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
    for label in (
        "Export Logs",
        "Data Cost Recovery",
        "Customers",
        "Proposed Cost Recovery",
        "Stations Registration",
    ):
        assert label not in base


def test_removed_templates_are_absent():
    for name in (
        "exports.html",
        "cost_recovery.html",
        "proposed_cost_recovery.html",
        "customers.html",
        "customer_edit.html",
        "stations.html",
    ):
        assert not (ROOT / "templates" / name).exists()


def test_removed_routes_are_not_registered_in_source():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    for route in (
        '@app.route("/exports")',
        '@app.route("/cost-recovery"',
        '@app.route("/proposed-cost-recovery"',
        '@app.route("/customers"',
        '@app.route("/stations"',
    ):
        assert route not in source
