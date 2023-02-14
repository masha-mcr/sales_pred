import pandas as pd


def run_quality_check(**kwargs: pd.DataFrame) -> None:
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    if not all(df in kwargs.keys() for df in ['sales', 'items', 'shops', 'item_categories', 'test']):
        raise ValueError('Not enough dataframes provided')
    else:
        for name, df in kwargs.items():
            if df.isna().any().any():
                print(f'NA values in [{name}]')
            if df.duplicated().any():
                print(f'Duplicates in [{name}]')
                print(df[df.duplicated()])
        print(kwargs['sales']['date'].head())
        # sales quality
        if not pd.api.types.is_datetime64_dtype(kwargs['sales']['date'].dtype):
            print('[sales] Wrong date dtype')
        else:
            if (kwargs['sales']['date'].min().year != 2013 and kwargs['sales']['date'].min().month != 1 and
                    kwargs['sales']['date'].max().year != 2015 and kwargs['sales']['date'].max().month != 10):
                print('[sales] Incorrect date range')

        if kwargs['sales']['date_block_num'].nunique() != 34:
            print('[sales] Date blocks missing')

        print('[sales] item price statistics:')
        print(kwargs['sales']['item_price'].describe().loc[['min', 'max', 'mean', '50%']])
        print('[sales] item_cnt_day statistics:')
        print(kwargs['sales']['item_cnt_day'].describe().loc[['min', 'max', 'mean', '50%']])

        # items quality
        bad_name_items = kwargs['items']['item_name'].str.contains('^[\\\/^.*\[\]~!@#$%^&()_+={}|\:;“’<,>?฿]+.*',
                                                                  regex=True, na=False).sum()
        if bad_name_items > 0:
            print(f'[items] Item names contain special characters ({bad_name_items} cases)')

        # shops quality
        bad_name_shops = kwargs['shops']['shop_name'].str.contains('^[\\\/^.*\[\]~!@#$%^&()_+={}|\:;“’<,>?฿]+.*',
                                                                   regex=True, na=False).sum()
        if bad_name_shops > 0:
            print(f'[shops] Shop names contain special characters ({bad_name_shops} cases)')

        # item categories quality
        bad_name_categories = kwargs['item_categories']['item_category_name'].str.contains('^[\\\/^.*\[\]~!@#$%^&()_+={}|\:;“’<,>?฿]+.*',
                                                                   regex=True, na=False).sum()
        if bad_name_categories > 0:
            print(f'[item_categories] Category names contain special characters ({bad_name_shops} cases)')

        # joint keys check
        shop_ids_sales = set(kwargs['sales']['shop_id'])
        item_ids_sales = set(kwargs['sales']['item_id'])
        shop_ids_shops = set(kwargs['shops']['shop_id'])
        item_ids_items = set(kwargs['items']['item_id'])
        shop_ids_test = set(kwargs['test']['shop_id'])
        item_ids_test = set(kwargs['test']['item_id'])
        cat_ids_items = set(kwargs['items']['item_category_id'])
        cat_ids_categories = set(kwargs['item_categories']['item_category_id'])
        if len(shop_ids_sales - shop_ids_shops) > 0:
            print('shops in [sales] not present in [shops]')
        if len(item_ids_sales - item_ids_items) > 0:
            print('items in [sales] not present in [items]')
        if len(shop_ids_test - shop_ids_shops) > 0:
            print('shops in [test] not present in [shops]')
        if len(item_ids_test - item_ids_items) > 0:
            print('items in [test] not present in [items]')
        if len(cat_ids_items - cat_ids_categories) > 0:
            print('categories in [items] not present in [item_categories]')








