import pyaedt

print("pyaedt version:", pyaedt.__version__)

desktop = pyaedt.Desktop(specified_version="2022.1", non_graphical=False, new_desktop_session=True)
print("AEDT launched successfully.")
print("AEDT process ID:", desktop.aedt_process_id)

desktop.release_desktop(close_projects=True, close_on_exit=True)
print("AEDT closed cleanly.")