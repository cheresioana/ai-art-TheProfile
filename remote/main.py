from gan.HandGan import generate_hand
import datetime

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    generate_hand()
    print("Timp de executie")
    print(datetime.datetime.now() - begin_time)