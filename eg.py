from datetime import datetime

def get_current_date_formatted():
    # 获取当前时间
    now = datetime.now()
    # 格式化为 YYYYMMDD 形式
    formatted_date = now.strftime('%Y%m%d-%H:%M:%S')
    return formatted_date

if __name__ == "__main__":
    current_date = get_current_date_formatted()
    print(f"当前日期的格式化形式是: {current_date}")