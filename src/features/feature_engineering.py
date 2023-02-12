import pandas as pd
import re
import os


class FeatureGenerator:
    def __init__(self, dest_dir: str, **datasets):
        self._dst = dest_dir
        os.mkdir(self._dst)

        if not all(df in datasets.keys() for df in ['items', 'shops', 'item_categories', 'sales']):
            raise ValueError('Not enough dataframes provided')

        self._shops = datasets['shops']
        # pd.read_csv(os.path.join(self._src, 'shops.csv'))
        self._items = datasets['items']
        # pd.read_csv(os.path.join(self._src, 'items.csv'))
        self._sales = datasets['sales']
        # pd.read_csv(os.path.join(self._src, 'sales_train.csv'), parse_dates=['date'], dayfirst=True)
        self._item_cat = datasets['item_categories']
        # pd.read_csv(os.path.join(self._src, 'item_categories.csv'))

    def saturate_shops(self) -> None:
        self._shops['city_name'] = self._shops['shop_name'].apply(
            lambda s: re.split(' (ТРЦ|ТЦ|ТК|ТРК|МТРЦ|\(|\"|ул)', s)[0])
        self._shops['city_name'].replace({'Якутск Орджоникидзе, 56': 'Якутск', 'Москва Магазин С21': 'Москва'},
                                         inplace=True)
        self._shops['city_id'] = self._shops['city_name'].rank(method='min').astype(int)

        shop_appearance = self._sales.groupby('shop_id').agg({'date': 'min'}).reset_index(). \
            rename(columns={'date': 'first_shop_appearance'})
        shops_saturated = self._shops.merge(shop_appearance, on='shop_id', how='left')
        shops_saturated.fillna(self._sales['date'].max(), inplace=True)

        shops_saturated.to_csv(os.path.join(self._dst, 'shops.csv'), index=False, date_format='%d.%m.%Y')

    def saturate_items(self) -> None:
        item_appearance = self._sales.groupby('item_id').agg({'date': 'min'}).reset_index(). \
            rename(columns={'date': 'first_item_appearance'})
        items_saturated = self._items.merge(item_appearance, on='item_id', how='left')
        items_saturated.fillna(self._sales['date'].max(), inplace=True)

        items_saturated.to_csv(os.path.join(self._dst, 'items.csv'), index=False, date_format='%d.%m.%Y')

    def saturate_categories(self) -> None:
        self._item_cat['item_global_category_name'] = self._item_cat['item_category_name']. \
            apply(lambda s: re.split(' (-|\()', s)[0])
        self._item_cat['item_global_category_id'] = self._item_cat['item_global_category_name'].rank(
            method='min').astype(int)

        self._item_cat.to_csv(os.path.join(self._dst, 'item_categories.csv'), index=False)

    def get_item_sales_lags(self) -> None:
        sales_monthly = self._sales.groupby(['date_block_num', 'item_id', 'shop_id']).agg({'item_cnt_day': 'sum'}). \
            reset_index().rename({'item_cnt_day': 'item_cnt_month'}, axis=1)
        sales_monthly_pivot = pd.pivot_table(sales_monthly,
                                             values='item_cnt_month',
                                             index=['item_id', 'shop_id'],
                                             columns=['date_block_num']).fillna(0).reset_index()

        sales_monthly_pivot.to_csv(os.path.join(self._dst, 'item_sales_monthly.csv'), index=False)

    def get_price_lags(self) -> None:
        price_monthly = self._sales.groupby(['date_block_num', 'item_id', 'shop_id']).agg(
            {'item_price': 'median'}).reset_index()
        price_monthly_pivot = pd.pivot_table(price_monthly, values='item_price', index=['item_id', 'shop_id'],
                                             columns=['date_block_num']).fillna(0).reset_index()

        price_monthly_pivot.to_csv(os.path.join(self._dst, 'item_price_monthly.csv'), index=False)

    def get_total_sales_lags(self) -> None:
        total_sales_monthly = self._sales.groupby(['date_block_num', 'shop_id']).agg(
            {'item_cnt_day': 'sum'}).reset_index().rename(
            columns={'item_cnt_day': 'item_sold_month'})
        total_sales_monthly_pivot = pd.pivot_table(total_sales_monthly, values='item_sold_month', index=['shop_id'],
                                                   columns=['date_block_num']).fillna(0).reset_index()
        total_sales_monthly_pivot.to_csv(os.path.join(self._dst, 'total_sales_monthly.csv'), index=False)

    def get_shop_variety(self) -> None:  # unique items sold in a shop per month
        shop_variety = self._sales.groupby(['shop_id', 'date_block_num']).agg({'item_id': 'nunique'}).reset_index()
        shop_variety.rename(columns={'item_id': 'shop_variety'}, inplace=True)
        shop_variety_pivot = pd.pivot_table(shop_variety, values='shop_variety', index=['shop_id'],
                                            columns=['date_block_num']).fillna(0).reset_index()
        shop_variety_pivot.to_csv(os.path.join(self._dst, 'shop_variety_monthly.csv'), index=False)

    def get_item_spread(self) -> None:  # unique shops that sell an item per month
        item_spread = self._sales.groupby(['item_id', 'date_block_num']).agg({'shop_id': 'nunique'}).reset_index()
        item_spread.rename(columns={'shop_id': 'item_spread'}, inplace=True)
        item_spread_pivot = pd.pivot_table(item_spread, values='item_spread', index=['item_id'],
                                           columns=['date_block_num']).fillna(0).reset_index()
        item_spread_pivot.to_csv(os.path.join(self._dst, 'item_spread_monthly.csv'), index=False)



