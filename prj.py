# timetable_csp.py
"""
Automated Timetable Generation as CSP - Basic implementation
Supports input from Excel (pandas) or SQLite (sqlite3).
Simple backtracking CSP solver with MRV and forward checking.
"""

import itertools
import random
import sqlite3
import sys
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

# Optional import for Excel reading
try:
    import pandas as pd
except Exception:
    pd = None
    # We'll handle if pandas is not available.


# --------------------------
# Data Classes
# --------------------------
@dataclass
class Course:
    id: str
    name: str
    credits: int
    type: str  # "Lecture" or "Lab"


@dataclass
class Instructor:
    id: str
    name: str
    preferred_slots: List[str]  # list of TimeSlot IDs
    qualified_courses: List[str]  # Course IDs


@dataclass
class Room:
    id: str
    type: str  # "Lab" or "Lecture"
    capacity: int


@dataclass
class TimeSlot:
    id: str  # e.g., "Mon_09_1030"
    day: str
    start: str
    end: str


@dataclass
class Section:
    id: str
    semester: int
    student_count: int


@dataclass
class LectureVar:
    course_id: str
    section_id: str
    lecture_idx: int  # which lecture number in week (0..k-1)
    var_id: str = field(init=False)

    def __post_init__(self):
        self.var_id = f"{self.course_id}__{self.section_id}__L{self.lecture_idx}"


# --------------------------
# Data Loader (Excel or SQLite)
# --------------------------
class DataLoader:
    def __init__(self, excel_path: Optional[str] = None, sqlite_path: Optional[str] = None):
        self.excel_path = excel_path
        self.sqlite_path = sqlite_path

        # containers
        self.courses: Dict[str, Course] = {}
        self.instructors: Dict[str, Instructor] = {}
        self.rooms: Dict[str, Room] = {}
        self.timeslots: Dict[str, TimeSlot] = {}
        self.sections: Dict[str, Section] = {}
        # course -> required lectures per week (assume 2 for lecture, 1 for lab unless provided)
        self.course_lectures: Dict[str, int] = {}

    def load(self):
        if self.excel_path and pd:
            self._load_from_excel(self.excel_path)
        elif self.sqlite_path:
            self._load_from_sqlite(self.sqlite_path)
        else:
            # fallback: generate sample data
            print("No data source provided or pandas not installed. Creating sample dataset.")
            self._create_sample_data()

    def _load_from_excel(self, path):
        x = pd.ExcelFile(path)
        # Expected sheets: Courses, Instructors, Rooms, TimeSlots, Sections, CourseLectures (optional)
        if "Courses" in x.sheet_names:
            df = pd.read_excel(x, "Courses")
            for _, r in df.iterrows():
                cid = str(r["CourseID"])
                self.courses[cid] = Course(
                    id=cid,
                    name=str(r.get("CourseName", cid)),
                    credits=int(r.get("Credits", 2)),
                    type=str(r.get("Type", "Lecture"))
                )
        if "Instructors" in x.sheet_names:
            df = pd.read_excel(x, "Instructors")
            for _, r in df.iterrows():
                iid = str(r["InstructorID"])
                prefs = []
                if "PreferredSlots" in r and pd.notna(r["PreferredSlots"]):
                    prefs = [s.strip() for s in str(r["PreferredSlots"]).split(";") if s.strip()]
                quals = []
                if "QualifiedCourses" in r and pd.notna(r["QualifiedCourses"]):
                    quals = [s.strip() for s in str(r["QualifiedCourses"]).split(";") if s.strip()]
                self.instructors[iid] = Instructor(iid, str(r.get("Name", iid)), prefs, quals)
        if "Rooms" in x.sheet_names:
            df = pd.read_excel(x, "Rooms")
            for _, r in df.iterrows():
                rid = str(r["RoomID"])
                self.rooms[rid] = Room(rid, str(r.get("Type", "Lecture")), int(r.get("Capacity", 30)))
        if "TimeSlots" in x.sheet_names:
            df = pd.read_excel(x, "TimeSlots")
            for _, r in df.iterrows():
                tid = str(r["TimeSlotID"]) if "TimeSlotID" in r else f"{r['Day']}_{r['StartTime']}_{r['EndTime']}"
                self.timeslots[tid] = TimeSlot(tid, str(r["Day"]), str(r["StartTime"]), str(r["EndTime"]))
        if "Sections" in x.sheet_names:
            df = pd.read_excel(x, "Sections")
            for _, r in df.iterrows():
                sid = str(r["SectionID"])
                self.sections[sid] = Section(sid, int(r.get("Semester", 1)), int(r.get("StudentCount", 30)))
        # optional mapping for lectures per course
        if "CourseLectures" in x.sheet_names:
            df = pd.read_excel(x, "CourseLectures")
            for _, r in df.iterrows():
                cid = str(r["CourseID"])
                self.course_lectures[cid] = int(r.get("LecturesPerWeek", 2))

    def _load_from_sqlite(self, path):
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        # Expect tables: Courses, Instructors, Rooms, TimeSlots, Sections, CourseLectures
        try:
            for row in cur.execute("SELECT CourseID, CourseName, Credits, Type FROM Courses"):
                cid, name, credits, typ = row
                self.courses[str(cid)] = Course(str(cid), str(name), int(credits), str(typ))
        except Exception:
            pass
        try:
            for row in cur.execute("SELECT InstructorID, Name, PreferredSlots, QualifiedCourses FROM Instructors"):
                iid, name, prefs, quals = row
                prefs_list = prefs.split(";") if prefs else []
                quals_list = quals.split(";") if quals else []
                self.instructors[str(iid)] = Instructor(str(iid), str(name), prefs_list, quals_list)
        except Exception:
            pass
        try:
            for row in cur.execute("SELECT RoomID, Type, Capacity FROM Rooms"):
                rid, typ, cap = row
                self.rooms[str(rid)] = Room(str(rid), str(typ), int(cap))
        except Exception:
            pass
        try:
            for row in cur.execute("SELECT TimeSlotID, Day, StartTime, EndTime FROM TimeSlots"):
                tid, day, st, et = row
                self.timeslots[str(tid)] = TimeSlot(str(tid), str(day), str(st), str(et))
        except Exception:
            pass
        try:
            for row in cur.execute("SELECT SectionID, Semester, StudentCount FROM Sections"):
                sid, sem, sc = row
                self.sections[str(sid)] = Section(str(sid), int(sem), int(sc))
        except Exception:
            pass
        try:
            for row in cur.execute("SELECT CourseID, LecturesPerWeek FROM CourseLectures"):
                cid, lpw = row
                self.course_lectures[str(cid)] = int(lpw)
        except Exception:
            pass
        conn.close()

    def _create_sample_data(self):
        # create small dataset for demonstration
        # Courses
        self.courses = {
            "CS101": Course("CS101", "Intro to CS", 3, "Lecture"),
            "CS102": Course("CS102", "Programming I", 3, "Lab"),
            "CS201": Course("CS201", "Data Structures", 3, "Lecture"),
            "CS202": Course("CS202", "Digital Logic", 3, "Lecture"),
        }
        # Course lectures per week
        self.course_lectures = {"CS101": 2, "CS102": 1, "CS201": 2, "CS202": 2}
        # Instructors
        self.instructors = {
            "I1": Instructor("I1", "Dr. A", ["Mon_09_1030", "Tue_09_1030"], ["CS101", "CS201"]),
            "I2": Instructor("I2", "Dr. B", ["Mon_1045_12"], ["CS102", "CS202"]),
            "I3": Instructor("I3", "Dr. C", [], ["CS101", "CS102", "CS201", "CS202"]),
        }
        # Rooms
        self.rooms = {
            "R101": Room("R101", "Lecture", 60),
            "LAB1": Room("LAB1", "Lab", 30),
            "R102": Room("R102", "Lecture", 40),
        }
        # Timeslots
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        slots = [("09", "10:30"), ("10:45", "12:15"), ("13", "14:30"), ("14:45", "16:15")]
        self.timeslots = {}
        for d in days:
            for sidx, (st, et) in enumerate(slots):
                tid = f"{d}_{st}_{et}".replace(":", "")
                self.timeslots[tid] = TimeSlot(tid, d, st, et)
        # Sections
        self.sections = {
            "S1": Section("S1", 1, 50),
            "S2": Section("S2", 1, 30),
        }


# --------------------------
# CSP Model
# --------------------------
class TimetableCSP:
    def __init__(self, data: DataLoader):
        self.data = data
        self.variables: List[LectureVar] = []
        self.domains: Dict[str, List[Tuple[str, str, str]]] = {}  # var_id -> list of (timeslot_id, room_id, instructor_id)
        # assignments: var_id -> (timeslot, room, instructor)
        self.assignments: Dict[str, Tuple[str, str, str]] = {}
        # convenience
        self.timeslots = list(self.data.timeslots.keys())
        self.rooms = list(self.data.rooms.keys())
        self.instructors = list(self.data.instructors.keys())

        self._create_variables_and_domains()

    def _create_variables_and_domains(self):
        # For each course & each section, create the required number of lecture variables.
        for course_id, course in self.data.courses.items():
            lectures_needed = self.data.course_lectures.get(course_id, 2 if course.type == "Lecture" else 1)
            for section_id in self.data.sections.keys():
                # Option: skip section-course mapping; here we assume every course offered to every section.
                # A real dataset should include which sections take which courses.
                for i in range(lectures_needed):
                    lv = LectureVar(course_id, section_id, i)
                    self.variables.append(lv)

        # Build domains for each variable
        for var in self.variables:
            domain = []
            course = self.data.courses[var.course_id]
            for t in self.timeslots:
                for r in self.rooms:
                    room = self.data.rooms[r]
                    # room type must match course type (hard constraint)
                    if course.type == "Lab" and room.type != "Lab":
                        continue
                    if course.type == "Lecture" and room.type == "Lab":
                        # allow lectures in lecture rooms only (simple policy)
                        continue
                    # instructor must be qualified
                    for inst_id, inst in self.data.instructors.items():
                        if var.course_id not in inst.qualified_courses:
                            continue
                        # capacity check
                        section = self.data.sections[var.section_id]
                        if room.capacity < section.student_count:
                            continue
                        domain.append((t, r, inst_id))
            # if domain empty (no match), we must relax: include any room of right type and any qualified instructor and timeslot
            if not domain:
                for t in self.timeslots:
                    for r in self.rooms:
                        room = self.data.rooms[r]
                        if course.type == "Lab" and room.type != "Lab":
                            continue
                        if course.type == "Lecture" and room.type == "Lab":
                            continue
                        for inst_id, inst in self.data.instructors.items():
                            if var.course_id not in inst.qualified_courses:
                                continue
                            domain.append((t, r, inst_id))
            self.domains[var.var_id] = domain

    # --------------------------
    # Constraint checks
    # --------------------------
    def check_hard(self, var_id: str, value: Tuple[str, str, str], assignments: Dict[str, Tuple[str, str, str]]) -> bool:
        # Hard constraints:
        # 1) No professor teaches more than one class at same timeslot
        # 2) No room used by >1 class same timeslot
        # 3) (Implicit) each var exists; required lectures per week were created
        # 4) Room type matched earlier when building domains
        t, r, inst = value
        for other_var, (ot, oroom, oinst) in assignments.items():
            if ot == t:
                if oinst == inst:
                    return False
                if oroom == r:
                    return False
        return True

    # Soft constraints scoring (lower is better)
    def soft_penalty(self, full_assignments: Dict[str, Tuple[str, str, str]]) -> int:
        penalty = 0
        # Soft1: avoid gaps for students (minimize gaps per section per day)
        # We'll compute for each section-day the number of gaps between earliest and latest class
        # First map section -> day -> list of numeric slot indices
        day_slot_index = {}  # timeslot id -> sequential index per day (0..)
        # build mapping
        # get ordered timeslots per day
        day_map = defaultdict(list)
        for tid, ts in self.data.timeslots.items():
            day_map[ts.day].append((tid, ts.start))
        for day, lst in day_map.items():
            lst_sorted = sorted(lst, key=lambda x: x[1])
            for idx, (tid, _) in enumerate(lst_sorted):
                day_slot_index[tid] = idx
        sections_day_slots = defaultdict(lambda: defaultdict(list))
        for var_id, (t, _, _) in full_assignments.items():
            # var_id includes section id as second field
            try:
                _, section_id, _ = var_id.split("__")
            except ValueError:
                # fallback parse
                parts = var_id.split("__")
                section_id = parts[1] if len(parts) > 1 else "S1"
            ts_idx = day_slot_index.get(t, None)
            day = self.data.timeslots[t].day if t in self.data.timeslots else None
            if ts_idx is not None and day:
                sections_day_slots[section_id][day].append(ts_idx)
        for section, days in sections_day_slots.items():
            for day, slots in days.items():
                if not slots:
                    continue
                lo, hi = min(slots), max(slots)
                total_span = hi - lo + 1
                classes = len(slots)
                gaps = total_span - classes
                penalty += gaps  # each gap = 1 penalty

        # Soft2: avoid early morning or late evening slots
        # Consider start times before 09:00 or after 16:00 as penalized
        for var_id, (t, _, _) in full_assignments.items():
            start = self.data.timeslots[t].start
            # simplistic parse hours
            try:
                hour = int(start.split(":")[0]) if ":" in start else int(start)
            except Exception:
                hour = 9
            if hour < 9 or hour >= 16:
                penalty += 1

        # Soft3: avoid scheduling same instructor in consecutive distant rooms
        # We'll penalize if same instructor has back-to-back timeslots with rooms not same building (we don't have building data)
        # Use room id difference heuristic: if room ids are different and not both start with same prefix -> penalty
        inst_times = defaultdict(list)
        for var_id, (t, r, inst) in full_assignments.items():
            day = self.data.timeslots[t].day
            ts_idx = day_slot_index.get(t, 0)
            inst_times[(inst, day)].append((ts_idx, r))
        for key, lst in inst_times.items():
            lst_sorted = sorted(lst, key=lambda x: x[0])
            # check consecutive indices (diff=1) but rooms different -> penalty
            for i in range(1, len(lst_sorted)):
                prev_idx, prev_room = lst_sorted[i - 1]
                cur_idx, cur_room = lst_sorted[i]
                if cur_idx - prev_idx == 1 and prev_room != cur_room:
                    # penalty for travel
                    penalty += 1

        # Soft4: distribute classes evenly across the week for instructors (penalize clumps)
        inst_day_count = defaultdict(lambda: defaultdict(int))
        for var_id, (t, _, inst) in full_assignments.items():
            day = self.data.timeslots[t].day
            inst_day_count[inst][day] += 1
        for inst, days in inst_day_count.items():
            counts = list(days.values())
            if counts:
                # more uneven distribution => higher variance => penalty
                mx = max(counts)
                mn = min(counts)
                penalty += (mx - mn)
        return penalty

    # --------------------------
    # Solver: Backtracking with MRV + Forward Checking
    # --------------------------
    def solve(self, time_limit_seconds: Optional[int] = None, randomize: bool = True) -> Optional[Dict[str, Tuple[str, str, str]]]:
        # order variables
        unassigned = set(v.var_id for v in self.variables)
        domains = deepcopy(self.domains)

        # pre-check: any variable with empty domain -> fail
        for v in unassigned:
            if not domains.get(v):
                print(f"Warning: variable {v} has empty domain.")
                return None

        # helper functions
        def select_var():
            # MRV: variable with smallest remaining domain
            mrv = None
            best_size = 1_000_000
            for v in unassigned:
                sz = len(domains[v])
                if sz < best_size:
                    best_size = sz
                    mrv = v
            return mrv

        def order_values(var_id: str):
            vals = domains[var_id][:]
            # prefer instructor's preferred slots (lower index)
            # compute a score
            def score(val):
                t, r, inst = val
                inst_obj = self.data.instructors.get(inst)
                score = 0
                if inst_obj and t in inst_obj.preferred_slots:
                    score -= 5
                # prefer rooms with enough capacity (already ensured) but prefer bigger rooms? neutral
                return score
            vals.sort(key=score)
            if randomize:
                # small shuffle among equal-score items
                random.shuffle(vals)
            return vals

        # forward checking: reduce domains after assignment
        def forward_check(var_id: str, val: Tuple[str, str, str], local_domains: Dict[str, List[Tuple[str, str, str]]]):
            t, r, inst = val
            removed = {}
            for other in list(unassigned):
                if other == var_id:
                    continue
                new_dom = []
                removed_from_other = []
                for cand in local_domains[other]:
                    ot, oroom, oinst = cand
                    conflict = False
                    if ot == t:
                        # same timeslot -> cannot share instructor or room
                        if oinst == inst or oroom == r:
                            conflict = True
                    if not conflict:
                        new_dom.append(cand)
                    else:
                        removed_from_other.append(cand)
                if removed_from_other:
                    removed[other] = removed_from_other
                local_domains[other] = new_dom
                if not new_dom:
                    # dead end
                    return False, removed
            return True, removed

        # backtracking recursive
        best_solution = None
        best_penalty = 10**9

        sys.setrecursionlimit(10000)

        def backtrack():
            nonlocal best_solution, best_penalty
            if not unassigned:
                # full assignment
                # evaluate soft penalty
                penalty = self.soft_penalty(self.assignments)
                # accept if penalty better
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_solution = deepcopy(self.assignments)
                    print(f"Found complete solution with penalty {penalty}")
                return

            var = select_var()
            if var is None:
                return
            vals = order_values(var)
            # try each value
            for val in vals:
                if not self.check_hard(var, val, self.assignments):
                    continue
                # assign
                self.assignments[var] = val
                unassigned.remove(var)
                # copy domains for forward checking
                saved_domains = {}
                # perform forward check
                ok, removed = forward_check(var, val, domains)
                if ok:
                    backtrack()
                # restore
                for other, removed_list in removed.items():
                    # Adding removed_list back is not necessary because we changed domains by rebuilding new lists,
                    # but safe approach: reconstruct domain as union of current + removed
                    domains[other].extend(removed_list)
                # unassign
                unassigned.add(var)
                del self.assignments[var]
            # end for
            return

        # Start backtracking
        backtrack()
        return best_solution

    # --------------------------
    # Utility: pretty print solution
    # --------------------------
    def print_solution(self, sol: Dict[str, Tuple[str, str, str]]):
        if not sol:
            print("No solution found.")
            return
        print("Solution (var -> (timeslot, room, instructor)):")
        for var, (t, r, inst) in sol.items():
            ts = self.data.timeslots.get(t)
            ts_str = f"{ts.day} {ts.start}-{ts.end}" if ts else t
            print(f"{var} -> {ts_str}, Room={r}, Inst={inst}")
        penalty = self.soft_penalty(sol)
        print(f"Soft penalty = {penalty}")
        # provide per-section summary
        per_section = defaultdict(list)
        for var, (t, r, inst) in sol.items():
            try:
                _, section_id, _ = var.split("__")
            except Exception:
                section_id = "S?"
            per_section[section_id].append((t, r, inst))
        for s, lst in per_section.items():
            print(f"Section {s} has {len(lst)} sessions scheduled.")


# --------------------------
# Example: Create sample sqlite DB (optional)
# --------------------------
def create_sample_sqlite(path="sample_timetable.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # Create tables
    cur.execute("CREATE TABLE IF NOT EXISTS Courses (CourseID TEXT PRIMARY KEY, CourseName TEXT, Credits INTEGER, Type TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS Instructors (InstructorID TEXT PRIMARY KEY, Name TEXT, PreferredSlots TEXT, QualifiedCourses TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS Rooms (RoomID TEXT PRIMARY KEY, Type TEXT, Capacity INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS TimeSlots (TimeSlotID TEXT PRIMARY KEY, Day TEXT, StartTime TEXT, EndTime TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS Sections (SectionID TEXT PRIMARY KEY, Semester INTEGER, StudentCount INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS CourseLectures (CourseID TEXT PRIMARY KEY, LecturesPerWeek INTEGER)")
    # insert some sample rows (if empty)
    cur.execute("SELECT COUNT(*) FROM Courses")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO Courses VALUES (?, ?, ?, ?)", [
            ("CS101", "Intro to CS", 3, "Lecture"),
            ("CS102", "Programming I", 3, "Lab"),
            ("CS201", "Data Structures", 3, "Lecture"),
            ("CS202", "Digital Logic", 3, "Lecture"),
        ])
    cur.execute("SELECT COUNT(*) FROM Instructors")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO Instructors VALUES (?, ?, ?, ?)", [
            ("I1", "Dr. A", "Mon_09_1030;Tue_09_1030", "CS101;CS201"),
            ("I2", "Dr. B", "Mon_1045_12", "CS102;CS202"),
            ("I3", "Dr. C", "", "CS101;CS102;CS201;CS202"),
        ])
    cur.execute("SELECT COUNT(*) FROM Rooms")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO Rooms VALUES (?, ?, ?)", [
            ("R101", "Lecture", 60),
            ("LAB1", "Lab", 30),
            ("R102", "Lecture", 40),
        ])
    cur.execute("SELECT COUNT(*) FROM TimeSlots")
    if cur.fetchone()[0] == 0:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        slots = [("09:00", "10:30"), ("10:45", "12:15"), ("13:00", "14:30"), ("14:45", "16:15")]
        for d in days:
            for st, et in slots:
                tid = f"{d}_{st}_{et}".replace(":", "")
                cur.execute("INSERT OR IGNORE INTO TimeSlots VALUES (?, ?, ?, ?)", (tid, d, st, et))
    cur.execute("SELECT COUNT(*) FROM Sections")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO Sections VALUES (?, ?, ?)", [("S1", 1, 50), ("S2", 1, 30)])
    cur.execute("SELECT COUNT(*) FROM CourseLectures")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO CourseLectures VALUES (?, ?)", [("CS101", 2), ("CS102", 1), ("CS201", 2), ("CS202", 2)])
    conn.commit()
    conn.close()
    print(f"Sample sqlite DB created at {path}")


# --------------------------
# Main / CLI
# --------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Timetable CSP solver - basic")
    parser.add_argument("--excel", help="Path to Excel file with sheets (Courses, Instructors, Rooms, TimeSlots, Sections, CourseLectures)", default=None)
    parser.add_argument("--sqlite", help="Path to sqlite db", default=None)
    parser.add_argument("--make-sample-db", help="Create sample sqlite DB at path", default=None)
    args = parser.parse_args()

    if args.make_sample_db:
        create_sample_sqlite(args.make_sample_db)
        return

    loader = DataLoader(excel_path=args.excel, sqlite_path=args.sqlite)
    loader.load()

    print("Loaded data summary:")
    print(f"Courses: {len(loader.courses)}, Instructors: {len(loader.instructors)}, Rooms: {len(loader.rooms)}, TimeSlots: {len(loader.timeslots)}, Sections: {len(loader.sections)}")

    csp = TimetableCSP(loader)
    print(f"Variables to assign: {len(csp.variables)}")

    solution = csp.solve()
    if solution:
        csp.print_solution(solution)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()
