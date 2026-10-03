"""Build an additive VP XML import from a verified official project export.

This script edits XML only. Import/export and teamwork commits must use Visual
Paradigm's official commands; preservation is verified on a separate VPP copy.
Existing IDs and creator metadata are retained. New objects omit audit metadata
so the application's normal creation path supplies it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main(source: Path, destination: Path) -> None:
    root = ET.parse(source).getroot()
    ns = root.tag.split("}")[0] + "}"
    ET.register_namespace("", ns[1:-1])
    models = {m.get("id"): m for m in root.iter(ns + "Model")}
    diagrams = {d.get("id"): d for d in root.iter(ns + "Diagram")}
    changed: set[str] = set()
    diagram_changes: set[str] = set()
    created: dict[str, str] = {}

    def uid(label):
        return "Elab" + hashlib.sha256(label.encode()).hexdigest()[:12]

    if uid("diagram.evidence.design") in diagrams:
        raise ValueError("This revision is already present; use the preserved pre-import export.")

    def props(obj):
        tag = "ModelProperties" if obj.tag == ns + "Model" else (
            "DiagramProperties" if obj.tag == ns + "Diagram" else "DiagramElementProperties"
        )
        return obj.find(ns + tag)

    def prop(obj, name, value=None, tag="StringProperty", ref=None):
        ps = props(obj)
        p = next((p for p in ps if p.get("name") == name), None)
        if p is None:
            p = ET.SubElement(ps, ns + tag, {"name": name, "displayName": name})
        for c in list(p):
            p.remove(c)
        p.attrib.pop("value", None)
        if value is not None:
            p.set("value", str(value))
        if ref:
            ET.SubElement(p, ns + "ModelRef", {"id": ref})
        return p

    def documentation(obj, text):
        rich = "<HTML><BODY><p>" + html.escape(text).replace("\n", "</p><p>") + "</p></BODY></HTML>"
        if obj.tag == ns + "Diagram":
            obj.set("documentation", text)
            obj.set("rtfDocumentation", rich)
            prop(obj, "documentation", text, "StringProperty")
            diagram_changes.add(obj.get("id"))
        else:
            d = prop(obj, "documentation", rich, "HTMLProperty")
            d.set("plainTextValue", text)
            changed.add(obj.get("id"))

    def rename(obj, name):
        obj.set("name", name)
        prop(obj, "name", name)
        changed.add(obj.get("id"))

    def clean_model(template, label, name, doc=""):
        m = copy.deepcopy(models[template])
        m.set("id", uid(label))
        for c in list(m):
            if c.tag != ns + "ModelProperties":
                m.remove(c)
        for p in list(props(m)):
            if p.get("name") in {"pmAuthor", "pmCreateDateTime", "pmLastModified", "masterView"}:
                props(m).remove(p)
            elif len(p):
                for c in list(p):
                    p.remove(c)
        rename(m, name)
        documentation(m, doc)
        models[m.get("id")] = m
        created[label] = m.get("id")
        return m

    def attach(parent, child):
        cm = parent.find(ns + "ChildModels")
        if cm is None:
            cm = ET.SubElement(parent, ns + "ChildModels")
        cm.append(child)
        changed.add(child.get("id"))

    def module(label, name, parent, doc, operations=(), attributes=()):
        m = clean_model("7OYM4bmAUAAADSYC", label, name, doc)
        attach(models[parent], m)
        for i, (op_name, signature, result) in enumerate(operations):
            op = clean_model("d6XN.wWAUAAADa.A", f"{label}.op.{i}", op_name, signature)
            rt = prop(op, "returnType", tag="TextModelProperty")
            ET.SubElement(rt, ns + "StringValue", {"value": result})
            prop(op, "static", "true", "BooleanProperty")
            attach(m, op)
        attr_template = next(x.get("id") for x in models.values() if x.get("modelType") == "Attribute")
        for i, (name_, typ, doc_) in enumerate(attributes):
            a = clean_model(attr_template, f"{label}.attr.{i}", name_, doc_)
            typ_prop = prop(a, "type", tag="TextModelProperty")
            ET.SubElement(typ_prop, ns + "StringValue", {"value": typ})
            attach(m, a)
        return m.get("id")

    def add_diagram(label, name, kind, parent, doc):
        source_id = "u03N.wWAUAAADbJ2" if kind == "ClassDiagram" else "WuPN.wWAUAAADbuO"
        d = copy.deepcopy(diagrams[source_id])
        d.set("id", uid(label)); d.set("name", name)
        for c in list(d):
            if c.tag != ns + "DiagramProperties": d.remove(c)
        for p in list(props(d)):
            if p.get("name") in {"pmAuthor", "pmCreateDateTime", "pmLastModified"}: props(d).remove(p)
        prop(d, "name", name)
        prop(d, "parentModel", tag="ModelRefProperty", ref=parent)
        prop(d, "_rootFrame", tag="ModelRefProperty")
        documentation(d, doc)
        ET.SubElement(d, ns + "Shapes"); ET.SubElement(d, ns + "Connectors")
        root.find(ns + "Diagrams").append(d)
        diagrams[d.get("id")] = d
        created[label] = d.get("id")
        owner = models[parent]
        sub = owner.find(ns + "SubDiagrams")
        if sub is None: sub = ET.SubElement(owner, ns + "SubDiagrams")
        ET.SubElement(sub, ns + "DiagramRef", {"id": d.get("id"), "name": name, "diagramType": kind})
        changed.add(parent)
        return d

    class_shape = next(s for s in diagrams["u03N.wWAUAAADbJ2"].iter(ns + "Shape") if s.get("model") == "QBAo4bmAUAAADR5m")
    life_shape = next(s for s in diagrams["WuPN.wWAUAAADbuO"].iter(ns + "Shape") if s.get("shapeType") == "InteractionLifeLine")

    def shape(d, label, model_id, x, y, w, h, life=False, frame=None):
        s = copy.deepcopy(life_shape if life else class_shape)
        s.set("id", uid(label)); s.set("model", model_id); s.set("name", models[model_id].get("name"))
        for c in list(s):
            if c.tag == ns + "ChildShapes": s.remove(c)
        for key, value in {"x": x, "y": y, "width": w, "height": h, "zorder": 2}.items():
            s.set(key, str(value)); prop(s, "zOrder" if key == "zorder" else key, value, "IntegerProperty")
        prop(s, "metaModelElement", tag="ModelRefProperty", ref=model_id)
        if frame: prop(s, "parentFrame", tag="ModelRefProperty", ref=frame)
        cap = s.find(ns + "Caption")
        cap.set("x", "0"); cap.set("y", "0"); cap.set("width", str(w)); cap.set("height", "35")
        prop(s, "requestFitSize", "false", "BooleanProperty")
        prop(s, "showOwnerOption", "0", "IntegerProperty")
        if not life and model_id.startswith("Elab"):
            # Full parameter contracts remain in operation documentation; this
            # overview displays operation names without implying empty inputs.
            prop(s, "showOperationSignature", "false", "BooleanProperty")
        d.find(ns + "Shapes").append(s)
        return s

    dependency_template = next(m.get("id") for m in models.values() if m.get("modelType") == "Dependency")
    dep_connector = next(c for d in diagrams.values() for c in d.iter(ns + "Connector") if c.get("shapeType") == "Dependency")
    msg_connector = next(c for c in diagrams["WuPN.wWAUAAADbuO"].iter(ns + "Connector") if c.get("shapeType") == "Message")

    def connector(d, label, model_id, name, a, b, y=None, message=False):
        c = copy.deepcopy(msg_connector if message else dep_connector)
        c.set("id", uid(label)); c.set("model", model_id); c.set("name", name)
        c.set("from", a.get("id")); c.set("to", b.get("id"))
        x1 = int(a.get("x")) + int(a.get("width")) // 2
        x2 = int(b.get("x")) + int(b.get("width")) // 2
        y1 = y if y is not None else int(a.get("y")) + int(a.get("height")) // 2
        y2 = y if y is not None else int(b.get("y")) + int(b.get("height")) // 2
        if y is None:
            if abs(x2-x1) > abs(y2-y1):
                x1 += (1 if x2>x1 else -1)*int(a.get("width"))//2
                x2 -= (1 if x2>x1 else -1)*int(b.get("width"))//2
            else:
                y1 += (1 if y2>y1 else -1)*int(a.get("height"))//2
                y2 -= (1 if y2>y1 else -1)*int(b.get("height"))//2
        for k, v in {"x": min(x1, x2)-100, "y": min(y1, y2)-100, "width": abs(x2-x1)+200, "height": abs(y2-y1)+200}.items():
            c.set(k, str(v)); prop(c, k, v, "IntegerProperty")
        prop(c, "metaModelElement", tag="ModelRefProperty", ref=model_id)
        points = c.find(ns + "Points")
        if points is None: points = ET.SubElement(c, ns + "Points")
        points.clear()
        ET.SubElement(points, ns + "Point", {"x": str(x1), "y": str(y1)})
        ET.SubElement(points, ns + "Point", {"x": str(x2), "y": str(y2)})
        cap = c.find(ns + "Caption")
        cap.set("x", str(min(x1, x2)+4)); cap.set("y", str((y1+y2)//2-20))
        cap.set("width", str(max(abs(x2-x1)-8, 180))); cap.set("height", "18")
        d.find(ns + "Connectors").append(c)
        return c

    def dependency(d, label, name, a, b):
        m = clean_model(dependency_template, label, name)
        prop(m, "from", tag="ModelRefProperty", ref=a.get("model"))
        prop(m, "to", tag="ModelRefProperty", ref=b.get("model"))
        attach(models["WlLCa1mGAqAALh.o"], m)
        connector(d, label+".shape", m.get("id"), name, a, b)

    uplift_doc = "SCRUM-475. Historical modeled Top-K comparison, not causal or held-out performance. Team baseline uses SUM(PTS)/SUM(POSS) over all team-season offensive rows. Selected modeled PPP uses SUM(PPP_PRED*POSS_OFF)/SUM(POSS_OFF). JSON and CSV contain scope, numerators, denominators, K, units, formulas, source SHA-256 and row contributions. Missing PTS fails explicitly; zero baseline gives null relative uplift. SCRUM-474 remains a separate evaluation dependency."
    cal_doc = "SCRUM-476. Expanding-window held-out-season PPP diagnostics using fixed Ridge and training-only StandardScaler/reliability normalization. Report signed bias, MAE/RMSE, continuous prediction bins, sparse-bin warnings and fold provenance. Same-season explanatory features mean retrospective evaluation, not pre-game forecasting proof."
    auth_doc = "Production authorization validates ES256/RS256 JWT signatures against Supabase JWKS plus issuer, audience and expiry; verified subject identifies the user. The current protected public.profiles.role authorizes resources, never user_metadata.role. Unassigned accounts remain pending. Analytics requires analyst; invalid token 401 and forbidden role 403."
    up_ep = module("uplift.endpoints", "TopKUpliftEndpoints", "Pq3mGLmAUAAADRS5", uplift_doc, [("create_topk_uplift_router", "rec: BaselineRecommender; source_path: Path. GET /metrics/topk-uplift[.json|.csv], require_role(analytics).", "APIRouter")])
    up_calc = module("uplift.calculation", "TopKUplift", "NIGWGLmAUAAADRT7", uplift_doc, [("compute_topk_uplift", "team_df, rankings, season, our_team, opp_team, k, w_off, provenance", "UpliftReport")])
    cal_ep = module("calibration.endpoints", "CalibrationEndpoints", "Pq3mGLmAUAAADRS5", cal_doc, [("create_calibration_router", "team_df, league_df. GET /metrics/calibration; analyst resource; fold/bin cache.", "APIRouter")])
    cal_calc = module("calibration.calculation", "Calibration", "NIGWGLmAUAAADRT7", cal_doc, [("compute_calibration", "team_df, league_df, n_splits, n_bins; fit only prior seasons.", "CalibrationReport"), ("summarize_calibration", "held-out prediction rows; continuous PPP bins.", "CalibrationSummary")])
    panel = module("calibration.panel", "CalibrationPanel", "7gDmGLmAUAAADRSj", "SCRUM-476. ModelMetricsPageClient composes this panel. Service/infrastructure fetch obtains Supabase session Bearer token. Loading, exact-value table and chart, sparse-bin warning, error/retry and aborted stale request handling.", [("render", "nSplits; report/loading/error state; /metrics/calibration.", "ReactElement")])
    design = add_diagram("diagram.evidence.design", "Elaboration - Analyst Evidence Design (475, 476)", "ClassDiagram", "7sZ6GLmAUAAADRP1", uplift_doc+"\n"+cal_doc+"\n"+auth_doc)
    shapes = {}
    for label, mid, x, y, w, h in [("metrics", "QBAo4bmAUAAADR5m",40,40,320,170),("panel",panel,450,40,350,170),("auth","jBg44bmAUAAADSDJ",900,40,320,170),("up_ep",up_ep,40,330,350,130),("cal_ep",cal_ep,450,330,350,130),("profile","aTt44bmAUAAADSFB",900,330,320,150),("up_calc",up_calc,40,650,350,140),("cal_calc",cal_calc,450,650,350,140)]:
        shapes[label] = shape(design,"evidence.shape."+label,mid,x,y,w,h)
    for a,b,name in [("metrics","panel","renders"),("panel","cal_ep","Bearer HTTP"),("up_ep","up_calc","computes"),("cal_ep","cal_calc","evaluates"),("auth","profile","current protected role")]: dependency(design,"evidence.dep."+a+b,name,shapes[a],shapes[b])
    documentation(models["QBAo4bmAUAAADR5m"], "Model metrics page with pipeline, holdout and tuning views. SCRUM-476 adds CalibrationPanel and its signed-bias, MAE/RMSE, binned PPP and fold-provenance view. "+cal_doc)
    documentation(models["aTt44bmAUAAADSFB"], auth_doc)
    documentation(models["Ub9lX1mGAqAALhoz"], uplift_doc+" Analyst selects season, teams, K and offense weight, then reads or downloads equivalent JSON/CSV evidence. Invalid filters 400/422; no eligible observations 404. Authorization failures 401/403.")

    state_doc = "SCRUM-481. public.gameplan_notes composite PK(user_id,season,our_team,opp_team). FK user_id -> auth.users ON DELETE CASCADE. Owner plus current protected profiles.role=coach is required for every CRUD action. Anonymous, analyst and pending users denied. Payload <=65536 bytes. Identity and server audit fields cannot be updated by clients. UI remains localStorage; cloud persistence is future SCRUM-482. Source: supabase/migrations/20261003211500_gameplan_state_schema.sql."
    state = module("gameplan.state", "GameplanState", "7sZ6GLmAUAAADRP1", state_doc, [], [("user_id", "uuid", "PK; auth.uid owner and FK auth.users.id"),("season", "text", "Composite PK; YYYY-YY season"),("our_team", "text", "Composite PK; uppercase team abbreviation"),("opp_team", "text", "Composite PK; different opponent; directed matchup"),("notes", "jsonb", "Object <=100 entries; key1..160; value<=4000 chars"),("plan", "jsonb", "Ordered <=50 unique {id,label}; no extra attributes"),("roles", "jsonb", "Exactly ballHandler,screener,cornerSpacer,cutter,safety; string<=120"),("revision", "bigint", "Starts1; server increments on update"),("created_at", "timestamptz", "Server-managed creation time"),("updated_at", "timestamptz", "Server-managed update time")])
    users = module("gameplan.owner", "AuthUser", "7sZ6GLmAUAAADRP1", "Existing Supabase auth.users identity. Deleting the owner cascades to private Gameplan state.", [], [("id", "uuid", "Primary key")])
    profile = module("gameplan.profile", "ProtectedProfile", "7sZ6GLmAUAAADRP1", "Existing public.profiles. Current role is administrator-controlled; user metadata is not authoritative.", [], [("id", "uuid", "Authenticated user ID"),("role", "coach | analyst | null", "Current protected authorization role; null means pending")])
    constraints = module("gameplan.guards", "GameplanStateGuards", "7sZ6GLmAUAAADRP1", state_doc, [("valid_gameplan_state", "Pure JSON validation; fixed keys, limits, unique plan IDs", "boolean"),("touch_gameplan_state", "Server timestamp and revision update trigger", "trigger"),("owner_coach_rls", "auth.uid=user_id AND current profiles.role=coach", "policy")])
    storage = add_diagram("diagram.gameplan.design", "Elaboration - Coach Gameplan Storage (481)", "ClassDiagram", "7sZ6GLmAUAAADRP1", state_doc)
    ss = {"state":shape(storage,"gameplan.shape.state",state,550,80,430,300),"users":shape(storage,"gameplan.shape.user",users,40,80,250,120),"profile":shape(storage,"gameplan.shape.profile",profile,1150,80,340,160),"guards":shape(storage,"gameplan.shape.guards",constraints,550,500,430,170)}
    for a,b,name in [("state","users","owner FK / cascade"),("state","profile","coach authorization"),("state","guards","validated and versioned")]: dependency(storage,"gameplan.dep."+a+b,name,ss[a],ss[b])

    def sequence(label, name, doc, participants, steps):
        owner = "kOHnKVmAUUE.bzDc"
        frame = clean_model("2uPN.wWAUAAADbuQ",label+".frame",name,doc)
        attach(models[owner],frame)
        d = add_diagram(label,name,"InteractionDiagram",owner,doc)
        prop(d,"_rootFrame",tag="ModelRefProperty",ref=frame.get("id"))
        participants_by_key = {}
        for i,(key,title,classifier) in enumerate(participants):
            life = clean_model("DuPN.wWAUAAADbuU",label+".life."+key,title)
            if classifier:prop(life,"baseClassifier",tag="TextModelProperty",ref=classifier)
            attach(frame,life)
            s = shape(d,label+".shape."+key,life.get("id"),40+i*270,40,210,180+len(steps)*65,True,frame.get("id"))
            participants_by_key[key] = s
        for i,(a,b,title) in enumerate(steps):
            m = clean_model("9JPN.wWAUAAADbxP",label+".message."+str(i),title)
            prop(m,"endRelationshipFromMetaModelElement",tag="ModelRefProperty",ref=participants_by_key[a].get("model"))
            prop(m,"endRelationshipToMetaModelElement",tag="ModelRefProperty",ref=participants_by_key[b].get("model"))
            prop(m,"sequenceNumber",str(i+1))
            action = prop(m,"actionType",tag="ModelProperty")
            # Copy the vendor's call action type rather than inventing message semantics.
            is_reply = title.startswith(("current role", "authorized", "weighted", "JSON/CSV", "report JSON", "PPP bins", "held-out fold"))
            action_type = "ActionTypeReturn" if is_reply else "ActionTypeCall"
            call = next(x for x in models.values() if x.get("modelType")==action_type)
            call_copy = clean_model(call.get("id"),label+".action."+str(i),"Call")
            action.append(call_copy)
            for tag,target in [("FromEnd",a),("ToEnd",b)]:
                end_template = next(x.get("id") for x in models.values() if x.get("modelType")=="MessageEnd")
                end = clean_model(end_template,label+".end."+str(i)+tag,"")
                prop(end,"EndModelElement",tag="ModelRefProperty",ref=participants_by_key[target].get("model"))
                ET.SubElement(m,ns+tag).append(end)
            attach(models["BwYKwNmGAqAALne6"],m)
            connector(d,label+".connector."+str(i),m.get("id"),title,participants_by_key[a],participants_by_key[b],150+i*65,True)
        return d

    sequence("diagram.uplift.sequence","Detailed Sequence - Analyst Top-K Uplift (475)",uplift_doc+"\n"+auth_doc+"\nAlternatives: 401 invalid token; 403 forbidden role; 400/422 invalid filters; 404 no eligible data.",[("ui","Analyst / Export Client",None),("api","TopKUpliftEndpoints",up_ep),("auth","AuthDependency","jBg44bmAUAAADSDJ"),("profile","Supabase Profiles",profile),("domain","TopKUplift",up_calc)],[("ui","api","GET /metrics/topk-uplift[.json|.csv] + Bearer"),("api","auth","require_role(analytics); verify JWT via JWKS"),("auth","profile","SELECT protected role for verified subject"),("profile","auth","current role = analyst"),("auth","api","authorized analyst"),("api","domain","compute from baseline rankings; retain all team-season rows"),("domain","api","weighted Top-K PPP minus seasonal SUM(PTS)/SUM(POSS)"),("api","ui","JSON/CSV + numerators, scope, units, K, source SHA-256")])
    sequence("diagram.calibration.sequence","Detailed Sequence - Inspect PPP Calibration (476)",cal_doc+"\n"+auth_doc+"\nUI supports loading, error/retry and cancellation. Empty bins have null means; sparse bins warn.",[("page","ModelMetricsPageClient","QBAo4bmAUAAADR5m"),("panel","CalibrationPanel",panel),("api","CalibrationEndpoints",cal_ep),("auth","Protected role / Auth","jBg44bmAUAAADSDJ"),("domain","Calibration",cal_calc)],[("page","panel","render with requested fold count"),("panel","api","GET /metrics/calibration + session Bearer"),("api","auth","verify JWT and current profiles.role = analyst"),("auth","api","authorized (otherwise 401/403)"),("api","domain","cache miss: compute_calibration(folds, bins)"),("domain","api","held-out fold results (trained only on prior seasons)"),("domain","api","PPP bins, signed bias, MAE/RMSE, fold provenance"),("api","panel","report JSON (or explicit error)"),("panel","page","chart + exact values + sparse warnings; retry errors")])

    deploy_doc = "SCRUM-506. Frontend https://nbaplayranker-seven.vercel.app on Vercel Hobby, Node.js 22, manually deployed from canonical shared Git checkout. Cloud Run nba-playranker-api in northamerica-northeast1: Python3.11/Uvicorn application.api_coordination.app:app, 1CPU/2GiB, min0/max2, concurrency2, numeric threads1. HTTPS REST/JSON + Bearer and explicit frontend CORS. Supabase Free Auth/JWKS, protected profiles and coach-owned gameplan_notes RLS. Analyst uplift/calibration reports use packaged datasets and ephemeral instance cache. Browser Gameplan cloud persistence remains future SCRUM-482. CAD10 budget alerts and CAD7 Cloud Run cap have delayed enforcement, not an absolute total billing guarantee."
    documentation(diagrams["8W4D.wWAUAAADc8a"],deploy_doc+"\n"+auth_doc)
    for mid,desc in [("vNzWuLmAUAAADYjS","Node.js 22 on Vercel Hobby; manual CLI deployment."),("BYz6uLmAUAAADYbu",deploy_doc),("H4z6uLmAUAAADYcJ","FastAPI Strategy API includes analyst Top-K uplift JSON/CSV and PPP calibration endpoints. "+auth_doc),("g569.wWAUAAADcpP",state_doc),(".rL2uLmAUAAADYmM",auth_doc)]:
        if mid in models:documentation(models[mid],desc)
    # Visible deployment labels and a database artifact complement the detailed
    # architecture documentation, without moving or deleting existing shapes.
    for mid,name in [("vNzWuLmAUAAADYjS","Node.js 22 Runtime"),("BYz6uLmAUAAADYbu","Cloud Run: min0 / max2 / 2GiB"),(".rL2uLmAUAAADYmM","Profiles Table (protected roles)")]:
        rename(models[mid],name)
        for d in diagrams.values():
            hit=False
            for obj in d.iter():
                if obj.get("model")==mid:obj.set("name",name);hit=True
            if hit:diagram_changes.add(d.get("id"))
    deployment=diagrams["8W4D.wWAUAAADc8a"]
    artifact=clean_model(".rL2uLmAUAAADYmM","gameplan.artifact","Gameplan Notes Table (coach owner)",state_doc)
    # The storage artifact shares the existing deployment artifact's package.
    parent_map_now={c:p for p in root.iter() for c in p}
    existing=models[".rL2uLmAUAAADYmM"]
    parent_map_now[existing].append(artifact)
    changed.add(artifact.get("id"))
    artifact_template=next(s for s in deployment.iter(ns+"Shape") if s.get("model")==".rL2uLmAUAAADYmM")
    artifact_shape=copy.deepcopy(artifact_template)
    artifact_shape.set("id",uid("gameplan.artifact.shape"));artifact_shape.set("model",artifact.get("id"));artifact_shape.set("name",artifact.get("name"))
    for key,value in {"x":1620,"y":890,"width":280,"height":65}.items():
        artifact_shape.set(key,str(value));prop(artifact_shape,key,value,"IntegerProperty")
    prop(artifact_shape,"metaModelElement",tag="ModelRefProperty",ref=artifact.get("id"))
    deployment.find(ns+"Shapes").append(artifact_shape)
    database_shape=next(s for s in deployment.iter(ns+"Shape") if s.get("model")=="g569.wWAUAAADcpP")
    dependency(deployment,"gameplan.artifact.dep","RLS + validation triggers",artifact_shape,database_shape)
    # Correct the misleading decoder result labels while keeping existing IDs.
    for mid,name in [("9JPN.wWAUAAADbxP","verified signing key (JWKS)"),("YZPN.wWAUAAADbx2","verified claims (sub; role resolved from protected profile)")]:
        rename(models[mid],name)
        for d in diagrams.values():
            hit=False
            for obj in d.iter():
                if obj.get("model")==mid:obj.set("name",name);hit=True
            if hit:diagram_changes.add(d.get("id"))
    documentation(diagrams["WuPN.wWAUAAADbuO"],auth_doc+" Current explicit protected-profile authorization exchange is shown in Detailed Sequence - Analyst Top-K Uplift (475).")
    for did,doc in [("UK3N.wWAUAAADbL0",uplift_doc+"\n"+cal_doc),("VR3N.wWAUAAADbRG",uplift_doc+"\n"+cal_doc),("u03N.wWAUAAADbJ2",cal_doc),("2l3N.wWAUAAADbTK",state_doc)]:
        documentation(diagrams[did],diagrams[did].get("documentation","")+"\n"+doc)

    # Keep complete changed model subtrees plus ancestor containers. The vendor
    # importer requires this context to resolve diagram/model references. A more
    # aggressively pruned document skipped new classes and diagrams in testing.
    parent_map={c:p for p in root.iter() for c in p}
    keep=set(changed)
    for mid in list(changed):
        p=parent_map.get(models[mid])
        while p is not None:
            if p.tag==ns+"Model":keep.add(p.get("id"))
            p=parent_map.get(p)
    def prune(node, complete=False):
        for c in list(node):
            if c.tag==ns+"Model":
                if not complete and c.get("id") not in keep:node.remove(c)
                else:prune(c,complete or c.get("id") in changed)
            elif c.tag==ns+"Diagrams":
                for d in list(c):
                    if d.get("id") not in diagram_changes:c.remove(d)
            else:prune(c,complete)
    prune(root)
    ET.indent(root,space="  ")
    ET.ElementTree(root).write(destination,encoding="UTF-8",xml_declaration=True)
    destination.with_suffix(".manifest.json").write_text(json.dumps({"source_export_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"created_ids":created,"changed_diagrams":sorted(diagram_changes),"changed_model_ids":sorted(changed)},indent=2)+"\n")
    print(json.dumps({"output":str(destination),"new_objects":len(created),"changed_diagrams":len(diagram_changes)},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    main(args.source,args.destination)
