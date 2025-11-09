
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

def _s(val):
    # stringify with empty string for None
    return "" if val is None else str(val)

st.markdown("""
<style>
.likert-row { margin-bottom: 0.75rem; }
.likert-q { font-weight: 600; margin-bottom: 0.25rem; }
.likert-anchors { font-size: 0.85rem; color: #555; margin-top: -0.25rem; }
</style>
""", unsafe_allow_html=True)
def likert_row(question: str, left_anchor: str, right_anchor: str, options, key: str):
    # Automatically merge anchors into first/last options
    options = list(options)
    if options and isinstance(options[0], int):
        options[0] = f"{options[0]} {left_anchor}"
        options[-1] = f"{options[-1]} {right_anchor}"

    st.markdown(f"**{question}**")
    choice = st.radio(
        label=question,
        options=options,
        horizontal=True,
        index=None,           # ensure no default selection
        key=key,
        label_visibility="collapsed",
    )
    return choice
# ==============================
# Config
# ==============================
st.set_page_config(page_title="Annotation App", layout="centered")

DATA_PATH = os.environ.get("DATA_PATH", "items.csv")
ASSIGN_COUNT = int(os.environ.get("ASSIGN_COUNT", "3"))
RESULTS_FALLBACK_CSV = os.environ.get("RESULTS_FALLBACK_CSV", "/mnt/data/results.csv")

# ==============================
# Helpers
# ==============================
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # Expect at least: prompt, response_a, response_b
    needed = {"prompt", "response_a", "response_b"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if "topic" not in df.columns:
        df["topic"] = ""
    # Ensure stable item_id for assignment tracking
    if "item_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "item_id"})
    df["item_id"] = df["item_id"].astype(str)
    return df

def get_engine():
    """Create a SQLAlchemy engine if DATABASE_URL is set; else return None."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    try:
        from sqlalchemy import create_engine
        return create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        st.warning(f"Could not create DB engine: {e}")
        return None

def ensure_results_table(conn):
    from sqlalchemy import text
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS results_third (
          id UUID PRIMARY KEY,
          timestamp TEXT,
          item_id TEXT,
          prolific_pid TEXT,
          session_id TEXT,
          topic TEXT,
          user_prompt TEXT,
          response_a TEXT,
          response_b TEXT,
          user_choice TEXT,
          comments_pref TEXT,
          wellbeing_choice TEXT,
          comments_well TEXT,
          ai_freq TEXT,
          aias_life TEXT,
          aias_work TEXT,
          aias_future TEXT,
          aias_humanity TEXT,
          aias_attention TEXT,
          tipi_reserved TEXT,
          tipi_trusting TEXT,
          tipi_lazy TEXT,
          tipi_relaxed TEXT,
          tipi_few_artistic TEXT,
          tipi_outgoing TEXT,
          tipi_fault_finding TEXT,
          tipi_thorough TEXT,
          tipi_nervous TEXT,
          tipi_imagination TEXT
        );
    """))


def ensure_assignments_table(conn):
    """Create an assignments table to guarantee up to 3 annotators per item."""
    from sqlalchemy import text
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS assignments_third (
          item_id TEXT NOT NULL,
          prolific_pid TEXT NOT NULL,
          status TEXT NOT NULL,            -- 'assigned' or 'completed'
          assigned_at TEXT NOT NULL,
          completed_at TEXT,
          PRIMARY KEY (item_id, prolific_pid)
        );
    """))

def reclaim_expired_assignments(conn, expiry_minutes=120):
    """Optionally free very old uncompleted assignments back to the pool (best effort)."""
    # This is a no-op for many DBs without date math functions; do simple prune client-side
    pass

def get_or_create_user_assignments(df, k):
    """Return a DataFrame of k items assigned to this user, using DB if available, else CSV fallback.
    Ensures each item receives at most 3 completed annotations globally.
    """
    pid = st.session_state.prolific_pid or "anon"
    eng = get_engine()

    # Helper to pick from a pool of candidate item_ids
    def pick_item_ids(candidates, already_assigned_ids, k):
        remain = [iid for iid in candidates if iid not in already_assigned_ids]
        # Deterministic shuffle based on PID for stability
        start = abs(hash(pid)) % (len(remain) if remain else 1)
        ordered = remain[start:] + remain[:start]
        return ordered[:k]

    if eng is None:
        # CSV fallback
        assign_csv = os.environ.get("ASSIGNMENTS_FALLBACK_CSV", "/mnt/data/assignments.csv")
        if os.path.exists(assign_csv):
            amap = pd.read_csv(assign_csv)
        else:
            amap = pd.DataFrame(columns=["item_id","prolific_pid","status","assigned_at","completed_at"])

        # compute completed counts per item
        comp_counts = amap[amap["status"]=="completed"].groupby("item_id").size().to_dict()
        # user's in-flight assignments not yet completed
        user_open = amap[(amap["prolific_pid"]==pid) & (amap["status"]=="assigned")]
        user_completed = amap[(amap["prolific_pid"]==pid) & (amap["status"]=="completed")]
        already = set(pd.concat([user_open["item_id"], user_completed["item_id"]], ignore_index=True).astype(str))

        # pool: items with completed<3
        df_ids = df["item_id"].astype(str).tolist()
        pool = [iid for iid in df_ids if comp_counts.get(iid,0) < 3]

        need = max(0, k - len(user_open))
        new_ids = pick_item_ids(pool, already, need)

        # append new assigned rows
        if need > 0 and new_ids:
            new_rows = pd.DataFrame({
                "item_id": new_ids,
                "prolific_pid": [pid]*len(new_ids),
                "status": ["assigned"]*len(new_ids),
                "assigned_at": [datetime.utcnow().isoformat()]*len(new_ids),
                "completed_at": [None]*len(new_ids)
            })
            amap = pd.concat([amap, new_rows], ignore_index=True)
            amap.to_csv(assign_csv, index=False)

        current_ids = amap[(amap["prolific_pid"]==pid) & (amap["status"]=="assigned")]["item_id"].astype(str).tolist()
        return df[df["item_id"].isin(current_ids)].reset_index(drop=True)

    # DB path
    from sqlalchemy import text
    with eng.begin() as conn:
        ensure_assignments_table(conn)

        # Completed counts per item
        res = conn.execute(text("""
            SELECT item_id,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed
            FROM assignments_third
            GROUP BY item_id
        """)).fetchall()
        comp_counts = {str(r[0]): int(r[1] or 0) for r in res}

        # User current assignments
        res2 = conn.execute(text("""
            SELECT item_id, status FROM assignments_third
            WHERE prolific_pid = :pid
        """), {"pid": pid}).fetchall()
        already = {str(r[0]) for r in res2}
        user_open = [str(r[0]) for r in res2 if r[1]=="assigned"]

        # Pool: items with completed<3
        df_ids = df["item_id"].astype(str).tolist()
        pool = [iid for iid in df_ids if comp_counts.get(iid,0) < 3]

        need = max(0, k - len(user_open))
        new_ids = pick_item_ids(pool, already, need)

        for iid in new_ids:
            conn.execute(text("""
                INSERT INTO assignments_third (item_id, prolific_pid, status, assigned_at)
                VALUES (:item_id, :pid, 'assigned', :ts)
                ON CONFLICT (item_id, prolific_pid) DO NOTHING
            """), {"item_id": iid, "pid": pid, "ts": datetime.utcnow().isoformat()})

        # Return the user's open assigned rows
        res3 = conn.execute(text("""
            SELECT item_id FROM assignments_third
            WHERE prolific_pid = :pid AND status='assigned'
        """), {"pid": pid}).fetchall()
        current_ids = [str(r[0]) for r in res3]
        return df[df["item_id"].isin(current_ids)].reset_index(drop=True)

def mark_assignment_completed(item_id):
    """Mark (item_id, current user) assignment as completed in DB or CSV."""
    pid = st.session_state.prolific_pid or "anon"
    eng = get_engine()
    if eng is None:
        assign_csv = os.environ.get("ASSIGNMENTS_FALLBACK_CSV", "/mnt/data/assignments.csv")
        if os.path.exists(assign_csv):
            amap = pd.read_csv(assign_csv)
        else:
            amap = pd.DataFrame(columns=["item_id","prolific_pid","status","assigned_at","completed_at"])
        mask = (amap["item_id"].astype(str)==str(item_id)) & (amap["prolific_pid"]==pid) & (amap["status"]=="assigned")
        if mask.any():
            amap.loc[mask, "status"] = "completed"
            amap.loc[mask, "completed_at"] = datetime.utcnow().isoformat()
            amap.to_csv(assign_csv, index=False)
        return

    from sqlalchemy import text
    with eng.begin() as conn:
        ensure_assignments_table(conn)
        conn.execute(text("""
            UPDATE assignments_third
            SET status='completed', completed_at=:ts
            WHERE item_id=:item_id AND prolific_pid=:pid AND status='assigned'
        """), {"item_id": str(item_id), "pid": pid, "ts": datetime.utcnow().isoformat()})
def insert_result(row_dict):
    """Insert to DB if configured; else append to fallback CSV."""
    eng = get_engine()
    if eng is None:
        # Fallback to CSV
        try:
            df = pd.DataFrame([row_dict])
            header = not os.path.exists(RESULTS_FALLBACK_CSV)
            df.to_csv(RESULTS_FALLBACK_CSV, mode="a", index=False, header=header)
        except Exception as e:
            st.error(f"Failed to write fallback CSV: {e}")
        return

    try:
        from sqlalchemy import text
        with eng.begin() as conn:
            ensure_results_table(conn)
            conn.execute(text("""
                INSERT INTO results_third ( id, timestamp, item_id, prolific_pid, session_id, topic, user_prompt,
                  response_a, response_b, user_choice,
                  comments_pref,
                  wellbeing_choice, comments_well,
                  ai_freq, aias_life, aias_work, aias_future, aias_humanity, aias_attention,
                  tipi_reserved, tipi_trusting, tipi_lazy, tipi_relaxed, tipi_few_artistic,
                  tipi_outgoing, tipi_fault_finding, tipi_thorough, tipi_nervous, tipi_imagination
                ) VALUES ( :id, :timestamp, :item_id, :prolific_pid, :session_id, :topic, :user_prompt,
                  :response_a, :response_b, :user_choice,
                  :comments_pref,
                  :wellbeing_choice, :comments_well,
                  :ai_freq, :aias_life, :aias_work, :aias_future, :aias_humanity, :aias_attention,
                  :tipi_reserved, :tipi_trusting, :tipi_lazy, :tipi_relaxed, :tipi_few_artistic,
                  :tipi_outgoing, :tipi_fault_finding, :tipi_thorough, :tipi_nervous, :tipi_imagination
                )
            """), row_dict)
    except Exception as e:
        st.error(f"Database insert failed: {e}")

def require_all_filled(values_dict):
    for k, v in values_dict.items():
        if v in (None, "", []):
            return False, k
    return True, None

def init_state():
    defaults = {
        "page": "consent",
        "assigned_rows": None,
        "pref_index": 0,
        "well_index": 0,
        "pref_data": [],   # per item: choice, comments_pref
        "well_data": [],   # per item: wellbeing_choice, comments_well
        "consented": False,
        "prolific_pid": "",
        "session_id": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ==============================
# UI Pieces
# ==============================
def consent_page():
    st.title("Consent")
    
    st.markdown("## Consent to Participate")
    st.write("""
    **Study purpose.** You are invited to take part in a research study about how people evaluate AI-generated responses.

    **What you’ll do.** You’ll view several AI responses to user prompts, select preferred responses, and leave comments. Approx. 3–7 minutes.

    **Risks/benefits.** Minimal risk; some scenarios could be mildly sensitive. You may skip anything you’d prefer not to answer.

    **Compensation.** As listed on Prolific.

    **Voluntary participation.** Your participation is voluntary. You may stop at any time.

    **Data.** We collect your Prolific PID, your inputs, your selections/ratings, and timestamps. Data will be used for research and may be shared in aggregate/anonymized form.

    **Contact/IRB.** If you have any questions, concerns or complaints about this research, its procedures, risks and benefits, contact the Protocol Director, Myra Cheng (myra@cs.stanford.edu)
    """)

    pid = st.text_input("Prolific PID (required)", value=st.session_state.prolific_pid)
    sid = st.text_input("Session ID (optional)", value=st.session_state.session_id)
    agree = st.checkbox("I have read and agree to the above.")
    proceed = st.button("Continue")

    if proceed:
        if not pid:
            st.error("Please enter your Prolific PID.")
            return
        if not agree:
            st.error("You must agree to continue.")
            return
        st.session_state.consented = True
        st.session_state.prolific_pid = pid
        st.session_state.session_id = sid
        st.session_state.page = "instructions"
        st.rerun()

def instructions_page():
    st.header("Instructions")
    st.markdown("""
You will read **3 prompts** and see **2 candidate responses** from an AI model (A/B) for each. """)
    if st.button("Start"):
        st.session_state.page = "pref"
        st.rerun()

def pick_assignments(df):
    # Assign rows via shared assignments table to guarantee up to 3 annotators per item
    return get_or_create_user_assignments(df, ASSIGN_COUNT)

def pref_phase_page(df):
    # ensure assignments in session
    if st.session_state.assigned_rows is None:
        st.session_state.assigned_rows = pick_assignments(df)

    i = st.session_state.pref_index
    if i >= len(st.session_state.assigned_rows):
        st.session_state.page = "wellbeing"
        st.rerun()
        

    row = st.session_state.assigned_rows.iloc[i]
    st.subheader(f"Preference — Item {i+1} of {len(st.session_state.assigned_rows)}")
    # if row.get("topic", ""):
    #     st.caption(f"Topic: {row['topic']}")
    st.markdown(f"**User Prompt**: {row['prompt']}")

    with st.form(f"form_pref_{i}", clear_on_submit=False):
        A_txt = str(row["response_a"])
        B_txt = str(row["response_b"])

        choice = st.radio(
            "Which response do you prefer from an AI model?",
            options=[A_txt, B_txt],
            key=f"pref_choice_{i}",
            index=None
        )
        comment = st.text_area(
            "Comment:",
            key=f"pref_comment_{i}", height=120
        )

        submitted = st.form_submit_button("Save & Continue")

        if submitted:
            ok, missing = require_all_filled({
                "preferred": choice,
                "comment": comment.strip(),
            })
            if not ok:
                st.error("Please complete the preference choice and the required comment.")
                st.stop()

            mapping = {A_txt: "A", B_txt: "B"}
            preferred_letter = mapping.get(choice, None)

            st.session_state.pref_data.append({ "item_id": str(row["item_id"]), "topic": row.get("topic", ""),
                "user_prompt": row["prompt"],
                "response_a": A_txt,
                "response_b": B_txt,
                "user_choice": preferred_letter,
                "comments_pref": comment.strip(),
            })

            st.session_state.pref_index += 1
            if st.session_state.pref_index >= len(st.session_state.assigned_rows):
                st.session_state.page = "wellbeing"
                st.rerun()
            else:
                st.rerun()
            

def wellbeing_phase_page():
    i = st.session_state.well_index
    rows = st.session_state.assigned_rows
    if rows is None or len(rows) == 0:
        st.error("No assigned items.")
        st.stop()

    if i >= len(rows):
        st.session_state.page = "survey"
        st.rerun()

    row = rows.iloc[i]
    st.subheader(f"Wellbeing — Item {i+1} of {len(rows)}")
    st.caption(f"Now you will assess the same prompts and responses, but from the perspective of benefiting the user's wellbeing.")

    # if row.get("topic", ""):
        # st.caption(f"Topic: {row['topic']}")
    st.markdown(f"**User Prompt**: {row['prompt']}")

    with st.form(f"form_well_{i}", clear_on_submit=False):
        A_txt = str(row["response_a"])
        B_txt = str(row["response_b"])

        wellbeing = st.radio(
            "Which response is **most beneficial to the user's long-term wellbeing**?",
            options=[A_txt, B_txt],
            key=f"well_choice_{i}",
            index=None
        )
        comment = st.text_area(
            "Comments:",
            key=f"well_comment_{i}", height=120
        )

        submitted = st.form_submit_button("Save & Continue")

        if submitted:
            ok, missing = require_all_filled({
                "wellbeing": wellbeing,
                "comment": comment.strip(),
            })
            if not ok:
                st.error("Please select a wellbeing choice and enter the required comment.")
                st.stop()

            mapping = {A_txt: "A", B_txt: "B"}
            well_letter = mapping.get(wellbeing, None)

            st.session_state.well_data.append({
                "wellbeing_choice": well_letter,
                "comments_well": comment.strip(),
            })

            st.session_state.well_index += 1
            if st.session_state.well_index >= len(rows):
                st.session_state.page = "survey"
                st.rerun()
            else:
                st.rerun()
            

def scale_radio(label, key):
    # 1..10 scale as radio with explicit numeric options
    opts = [f"{i}" for i in range(1, 11)]
    return st.radio(label, options=opts, index=None, key=key, horizontal=True)

def survey_page():
    st.header("AI Use & Personality")
    st.markdown("Please answer all questions below.")
    ai_freq = st.radio(
        "How often do you use AI chatbots?",
        ["Daily", "A few times a week", "A few times a month", "Once every few months", "Never"],
        index=None,
    )
    # with st.form("form_survey"):
    st.subheader("AI Use")
    
    # --- Attitudes toward AI (1–10) with attention check
    st.subheader("How much do you agree with the following statements?")

    aias_life = likert_row(
        "I believe that AI will improve my life.",
        left_anchor="Not at all",
        right_anchor="Completely agree",
        options=range(1, 11),
        key="aias_life",
    )

    aias_work = likert_row(
        "I believe that AI will improve my work.",
        left_anchor="Not at all",
        right_anchor="Completely agree",
        options=range(1, 11),
        key="aias_work",
    )

    aias_future = likert_row(
        "I think I will use AI technology in the future.",
        left_anchor="Not at all",
        right_anchor="Completely agree",
        options=range(1, 11),
        key="aias_future",
    )

    aias_humanity = likert_row(
        "I think AI technology is positive for humanity.",
        left_anchor="Not at all",
        right_anchor="Completely agree",
        options=range(1, 11),
        key="aias_humanity",
    )

    aias_attention = likert_row(
        "As an attention check, please select 10 (Completely agree).",
        left_anchor="Not at all",
        right_anchor="Completely agree",
        options=range(1, 11),
        key="aias_attention",
    )

    # --- TIPI (1–5)
    st.subheader("I see myself as someone who...")

    tipi_reserved = likert_row("… is reserved",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_reserved")
    tipi_trusting = likert_row("… is generally trusting",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_trusting")
    tipi_lazy = likert_row("… tends to be lazy",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_lazy")
    tipi_relaxed = likert_row("… is relaxed, handles stress well",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_relaxed")
    tipi_few_artistic = likert_row("… has few artistic interests",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_few_artistic")
    tipi_outgoing = likert_row("… is outgoing, sociable",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_outgoing")
    tipi_fault_finding = likert_row("… tends to find fault with others",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_fault_finding")
    tipi_thorough = likert_row("… does a thorough job",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_thorough")
    tipi_nervous = likert_row("… gets nervous easily",
        "Disagree strongly", "Agree strongly", range(1, 6), "tipi_nervous")
    tipi_imagination = likert_row("… has an active imagination",
    "Disagree strongly", "Agree strongly", range(1, 6), "tipi_imagination")

    # submit_survey = st.form_submit_button("Submit All")

    if st.button("Submit"):
        # Validate
        req = {
            "ai_freq": ai_freq,
            "aias_life": aias_life, "aias_work": aias_work, "aias_future": aias_future,
            "aias_humanity": aias_humanity, "aias_attention": aias_attention,
            "tipi_reserved": tipi_reserved, "tipi_trusting": tipi_trusting, "tipi_lazy": tipi_lazy,
            "tipi_relaxed": tipi_relaxed, "tipi_few_artistic": tipi_few_artistic,
            "tipi_outgoing": tipi_outgoing, "tipi_fault_finding": tipi_fault_finding,
            "tipi_thorough": tipi_thorough, "tipi_nervous": tipi_nervous, "tipi_imagination": tipi_imagination
        }
        ok, missing = require_all_filled(req)
        if not ok:
            st.error("Please complete all AI-use and personality questions.")
            st.stop()

        # One-time write: merge pref_data and well_data per item and save
        rows_out = []
        N = len(st.session_state.assigned_rows)
        if len(st.session_state.pref_data) != N or len(st.session_state.well_data) != N:
            st.error("Internal error: missing items. Please refresh and restart.")
            st.stop()

        for idx in range(N):
            pref = st.session_state.pref_data[idx]
            well = st.session_state.well_data[idx]
            # Backfill item_id for older sessions
            if 'item_id' not in pref:
                try:
                    pref['item_id'] = str(st.session_state.assigned_rows.iloc[idx]['item_id'])
                except Exception:
                    pref['item_id'] = ''
            row = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "item_id": pref.get("item_id", ""),
                "item_id": pref["item_id"],
                "prolific_pid": st.session_state.prolific_pid,
                "session_id": st.session_state.session_id,
                "topic": pref.get("topic", ""),
                "user_prompt": pref["user_prompt"],
                "response_a": pref["response_a"],
                "response_b": pref["response_b"],
                "user_choice": pref["user_choice"],
                "comments_pref": pref["comments_pref"],
                "wellbeing_choice": well["wellbeing_choice"],
                "comments_well": well["comments_well"],
                # AI Use / Attitudes
                "ai_freq": ai_freq,
                "aias_life": _s(aias_life),
                "aias_work": _s(aias_work),
                "aias_future": _s(aias_future),
                "aias_humanity": _s(aias_humanity),
                "aias_attention": _s(aias_attention),
                # TIPI
                "tipi_reserved": _s(tipi_reserved),
                "tipi_trusting": _s(tipi_trusting),
                "tipi_lazy": _s(tipi_lazy),
                "tipi_relaxed": _s(tipi_relaxed),
                "tipi_few_artistic": _s(tipi_few_artistic),
                "tipi_outgoing": _s(tipi_outgoing),
                "tipi_fault_finding": _s(tipi_fault_finding),
                "tipi_thorough": _s(tipi_thorough),
                "tipi_nervous": _s(tipi_nervous),
                "tipi_imagination": _s(tipi_imagination),
            }
            rows_out.append(row)

        # Write out
        for r in rows_out:
            insert_result(r)
            mark_assignment_completed(r.get('item_id'))

        st.session_state.page = "thanks"
        
        

def thanks_page():
    st.success("✅ Thanks! Your responses have been recorded.")
    completion_code = os.environ.get("PROLIFIC_CODE", "C1I7QVTN")
    redirect_url = f"https://app.prolific.com/submissions/complete?cc={completion_code}"
    st.markdown(f"[Return to Prolific]({redirect_url})")
    st.markdown("You may now close this window.")

# ==============================
# App flow
# ==============================
def main():
    # Init state & data
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.assigned_rows = None
        st.session_state.pref_index = 0
        st.session_state.well_index = 0
        st.session_state.pref_data = []
        st.session_state.well_data = []
        st.session_state.consented = False
        st.session_state.prolific_pid = ""
        st.session_state.session_id = ""
        st.session_state.page = "consent"
    query_params = st.query_params
    st.session_state.prolific_pid = query_params.get("PROLIFIC_PID", ["anon"])
    st.session_state.session_id = query_params.get("SESSION_ID", ["none"])
    # st.session_state.prolific_pid = ""
    # st.session_state.session_id = ""

    df = load_dataset()

    page = st.session_state.page
    if page == "consent":
        consent_page()
    elif page == "instructions":
        instructions_page()
    elif page == "pref":
        pref_phase_page(df)
    elif page == "wellbeing":
        wellbeing_phase_page()
    elif page == "survey":
        survey_page()
    elif page == "thanks":
        thanks_page()
    else:
        st.session_state.page = "consent"
        st.rerun()
        

if __name__ == "__main__":
    main()
