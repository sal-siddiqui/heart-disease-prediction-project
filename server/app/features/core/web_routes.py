from flask import Blueprint, render_template


# initialize blueprint
bp_core_web = Blueprint(
    name="core_web",
    import_name=__name__,
    template_folder="../../templates/core/",
    url_prefix="/",
)


# ——— Demo Route
@bp_core_web.route("/demo")
def demo():
    return render_template("demo.html", title="Demo")


# ——— CSV Route
@bp_core_web.route("/csv")
def csv():
    return render_template("csv.html", title="CSV")


# ——— REST API Route
@bp_core_web.route("/rest_apis")
def rest_apis():
    return render_template("rest-apis.html", title="REST APIs")
