import datetime
import dateutil.relativedelta
import calendar


def get_now():
    now = datetime.datetime.now()
    return now


def get_today():
    today = datetime.date.today()
    return today


def get_yesterday():
    today = get_today()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday


def get_month():
    today = datetime.datetime.today()
    current_date = datetime.datetime(today.year, today.month, 1)
    month = current_date.strftime("%Y-%m")
    return month


def get_hour():
    hour = datetime.datetime.now().hour
    return hour


def get_last_monday():
    today = get_today()
    last_monday = today - datetime.timedelta(today.weekday())
    return last_monday


def get_next_monday():
    today = get_today()
    next_monday = today + datetime.timedelta(days=-today.weekday(), weeks=1)
    return next_monday


def get_previous_month():
    today = get_today()
    first = today.replace(day=1)
    lastMonth = first - datetime.timedelta(days=1)
    previous_month = lastMonth.strftime("%Y-%m")
    return previous_month


def get_previous_month_today():
    now = get_now()
    previous_month_today = now + dateutil.relativedelta.relativedelta(months=-1)
    previous_month_today = previous_month_today.strftime("%Y-%m-%d")
    return previous_month_today


def get_last_day_of_month(year, month):
    last_day_of_month = calendar.monthrange(year, month)[1]
    last_day_of_month = str(last_day_of_month)
    return last_day_of_month


if __name__ == "__main__":
    print(str(get_now()))
    print(str(get_today()))
    print(str(get_yesterday()))
    print(str(get_hour()))
    print(str(get_last_monday()))
    print(str(get_next_monday()))
    print(str(get_previous_month()))
    print(get_previous_month_today())

    print("get_last_day_of_month test")
    print(get_last_day_of_month(2020, 11))
    print(get_last_day_of_month(2020, 12))
    print(get_last_day_of_month(2021, int("01")))
    print(get_last_day_of_month(2021, int("02")))
    print(get_last_day_of_month(2021, int("03")))

    print("get_month test")
    print(get_month())

