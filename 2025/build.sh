pyinstaller --hidden-import='PIL._tkinter_finder' --onefile --nowindowed --noconfirm --icon=./robot.ico --exclude-module cost_fn --exclude-module PyQt5 main.py