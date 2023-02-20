import pandas as pd
import numpy as np
import os
import re
from datetime import date


def etl(src_path: os.PathLike, dst_path: os.PathLike) -> None:
    sales = pd.read_csv(os.path.join(src_path, 'sales_train.csv'), parse_dates=['date'], dayfirst=True)
    shops = pd.read_csv(os.path.join(src_path, 'shops.csv'))
    items = pd.read_csv(os.path.join(src_path, 'items.csv'))
    item_cat = pd.read_csv(os.path.join(src_path, 'item_categories.csv'))
    test = pd.read_csv(os.path.join(src_path, 'test.csv'))
    sales = sales[sales['item_price'] > 0]
    sales = sales[sales['item_cnt_day'] >= 0]

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


def resample_sales(sales, shops, items,
                   zeros_frac=0.1, big_targets_frac=0.1, big_target_thres=0.95, drop_ones_frac=0.5):

    def get_random_dates(n):
        years = np.random.choice([2013, 2015], size=n, replace=True)
        months = np.random.choice(np.arange(1, 13), size=n, replace=True)
        days = np.random.choice(np.arange(1, 29), size=n, replace=True)
        dates = list(map(lambda year, month, day: date(year, month, day), years, months, days))
        return dates

    possible_shop_item_pairs = shops.merge(items, how='cross')
    real_shop_item_pairs = set(zip(sales['shop_id'], sales['item_id']))
    artificial_zero_sales_pairs = np.array([*( possible_shop_item_pairs - real_shop_item_pairs)])

    price_per_category = sales.merge(items, on='item_id').groupby(['item_category_id']).agg(
        {'item_price': 'median'}).reset_index()

    zero_sales_n = int(sales.shape[0]*zeros_frac)
    zero_sales = pd.DataFrame()
    zero_sales['date'] = pd.to_datetime(get_random_dates(zero_sales_n), format="%Y-%m-%d")
    zero_sales['date_block_num'] = (zero_sales['date'].dt.year - 2013) * 12 + zero_sales[
        'date'].dt.month + 1
    zero_sales_pairs = artificial_zero_sales_pairs[np.random.randint(artificial_zero_sales_pairs.shape[0], size=zero_sales_n), :]
    zero_sales['shop_id'] = zero_sales_pairs[:, 0]
    zero_sales['item_id'] = zero_sales_pairs[:, 1]
    zero_sales = zero_sales.merge(items, on='item_id').merge(price_per_category, on='item_category_id')
    zero_sales['item_cnt_day'] = 0
    zero_sales = zero_sales[sales.columns]

    big_targets = sales[sales['item_cnt_day'] > sales['item_cnt_day'].quantile(big_target_thres)]
    big_target_shops = big_targets['shop_id'].unique()
    big_target_items = big_targets['item_id'].unique()
    big_target_targets = big_targets['item_cnt_day'].unique()

    big_sales_n = int(sales.shape[0]*big_targets_frac)
    big_sales = pd.DataFrame()
    big_sales['date'] = pd.to_datetime(get_random_dates(big_sales_n), format="%Y-%m-%d")
    big_sales['date_block_num'] = (big_sales['date'].dt.year - 2013) * 12 + big_sales['date'].dt.month + 1
    big_sales['shop_id'] = np.random.choice(big_target_shops, size=big_sales_n, replace=True)
    big_sales['item_id'] = np.random.choice(big_target_items, size=big_sales_n, replace=True)
    big_sales = big_sales.merge(items, on='item_id').merge(price_per_category, on='item_category_id')
    big_sales['item_cnt_day'] = np.random.choice(big_target_targets, size=big_sales_n, replace=True)
    big_sales = big_sales[sales.columns]

    new_sales = pd.concat([sales, zero_sales, big_sales])
    new_sales = new_sales.drop(new_sales[new_sales['item_cnt_day'] == 1.0].sample(frac=drop_ones_frac).index)

    return new_sales







