import pandas as pd
import os
import re


def etl(src_path: os.PathLike, dst_path: os.PathLike) -> None:
    sales = pd.read_csv(os.path.join(src_path, 'sales_train.csv'), parse_dates=['date'], dayfirst=True)
    shops = pd.read_csv(os.path.join(src_path, 'shops.csv'))
    items = pd.read_csv(os.path.join(src_path, 'items.csv'))
    item_cat = pd.read_csv(os.path.join(src_path, 'item_categories.csv'))
    test = pd.read_csv(os.path.join(src_path, 'test.csv'))
    sales = sales[sales['item_price'] > 0]
    sales = sales[sales['item_cnt_day'] > 0]

    items['item_name'] = items['item_name'].apply(lambda name: re.sub('^[\\\/^.*\[\]~!@#$%^&()_+={}|\:;“’<,>?฿]+', '', name))
    shops['shop_name'] = shops['shop_name'].apply(lambda name: re.sub('^[\\\/^.*\[\]~!@#$%^&()_+={}|\:;“’<,>?฿]+', '', name))

    shops_mapping = {10: 11, 40: 39, 0: 57, 1: 58}
    items_mapping = {12: 14690}

    sales['shop_id'].replace(shops_mapping, inplace=True)
    sales['item_id'].replace(items_mapping, inplace=True)

    test['item_id'].replace(items_mapping, inplace=True)
    test['shop_id'].replace(shops_mapping, inplace=True)

    sales = sales.groupby(['date', 'date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'median', 'item_cnt_day': 'sum'}).reset_index()

    os.mkdir(dst_path)
    sales.to_csv(os.path.join(dst_path, 'sales_train.csv'), index=False, date_format='%d.%m.%Y')
    shops.to_csv(os.path.join(dst_path, 'shops.csv'), index=False)
    items.to_csv(os.path.join(dst_path, 'items.csv'), index=False)
    item_cat.to_csv(os.path.join(dst_path, 'item_categories.csv'), index=False)
    test.to_csv(os.path.join(dst_path, 'test.csv'))







