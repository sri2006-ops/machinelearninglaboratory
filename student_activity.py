import re
from collections import defaultdict

# ---------- Student Class ----------
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def get_summary(self):
        logins = sum(1 for a in self.activities if a[0] == "LOGIN")
        submissions = sum(1 for a in self.activities if a[0] == "SUBMIT_ASSIGNMENT")
        return logins, submissions


# ---------- Generator to read log file ----------
def read_logs(filename):
    pattern = re.compile(
        r'^(S\d+)\s*\|\s*(\w+)\s*\|\s*(LOGIN|LOGOUT|SUBMIT_ASSIGNMENT)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{2}:\d{2})$'
    )

    with open(filename, 'r') as file:
        for line in file:
            try:
                line = line.strip()
                match = pattern.match(line)
                if not match:
                    raise ValueError
                yield match.groups()
            except:
                print("Invalid entry skipped:", line)


# ---------- Main Program ----------
students = {}
daily_stats = defaultdict(int)
login_tracker = defaultdict(int)

for sid, name, activity, date, time in read_logs("activity_log.txt"):

    if sid not in students:
        students[sid] = Student(sid, name)

    students[sid].add_activity(activity, date, time)
    daily_stats[(date, activity)] += 1

    # Detect abnormal behavior
    if activity == "LOGIN":
        login_tracker[sid] += 1
    elif activity == "LOGOUT":
        login_tracker[sid] = max(0, login_tracker[sid] - 1)


# ---------- Display and Save Report ----------
with open("final_report.txt", "w") as out:
    print("\nSTUDENT ACTIVITY REPORT\n")
    out.write("STUDENT ACTIVITY REPORT\n\n")

    for student in students.values():
        logins, submissions = student.get_summary()

        report = (
            f"Student ID: {student.student_id}\n"
            f"Name: {student.name}\n"
            f"Total Logins: {logins}\n"
            f"Total Submissions: {submissions}\n"
            "-------------------------\n"
        )

        print(report)
        out.write(report)

print("\nABNORMAL BEHAVIOR REPORT")
for sid, count in login_tracker.items():
    if count > 0:
        print(f"Multiple logins without logout detected for {sid}")

print("\nDAILY ACTIVITY STATISTICS")
for (date, activity), count in daily_stats.items():
    print(f"{date} - {activity}: {count}")
