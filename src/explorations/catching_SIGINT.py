import win32api
import time
import odl
import _thread



def handler(signum, hook=None):
    print(signum, hook)
    print("Custom Message!! :))")
    _thread.interrupt_main()
    return 1



win32api.SetConsoleCtrlHandler(handler, 1)

while True:
    print("Press Ctr+C")
    time.sleep(10)
