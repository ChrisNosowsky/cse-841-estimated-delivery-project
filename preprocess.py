# ==============================================================================
# Preprocessing Stage
#
# Authors: Christopher Nosowsky
#
# ==============================================================================

import pandas as pd
import numpy as np
import mpu
import datetime as dt
from constants import Constants
from uszipcode import SearchEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


class PreProcess:
    """
    Preprocess shipment records
    """

    def __init__(self, filename):
        self.filename = filename
        self.dataset = pd.read_csv(self.filename).head(500000).dropna(axis=0, how='any')
        self.features = None
        self.feature_names = None
        self.labels = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.handling_x_train = None
        self.handling_x_test = None
        self.shipping_x_train = None
        self.shipping_x_test = None
        self.handling_y_train = None
        self.handling_y_test = None
        self.shipping_y_train = None
        self.shipping_y_test = None
        self.b2c_c2c = None
        self.seller_id = None
        self.declared_handling_days = None
        self.acceptance_scan_timestamp = None
        self.shipment_method_id = None
        self.shipping_fee = None
        self.carrier_min_estimate = None
        self.carrier_max_estimate = None
        self.item_zip = None
        self.buyer_zip = None
        self.category_id = None
        self.item_price = None
        self.quantity = None
        self.payment_datetime = None
        self.delivery_date = None
        self.weight = None
        self.weight_units = None
        self.package_size = None

    def pre_process(self):
        """
        Preprocesses the Dataset provided by the class.
        Fills in missing data, extracts features and labels, and encodes categorial values
        :return:
        """
        self.b2c_c2c = np.array(self.dataset[Constants.B2C_C2C])
        self.seller_id = np.array(self.dataset[Constants.SELLER_ID])
        self.declared_handling_days = np.array(self.dataset[Constants.DECLARED_HANDLING_DAYS])
        self.acceptance_scan_timestamp = np.array(self.dataset[Constants.ACCEPTANCE_SCAN_TIMESTAMP])
        self.shipment_method_id = np.array(self.dataset[Constants.SHIPMENT_METHOD_ID])
        self.shipping_fee = np.array(self.dataset[Constants.SHIPPING_FEE])
        self.carrier_min_estimate = np.array(self.dataset[Constants.CARRIER_MIN_ESTIMATE])
        self.carrier_max_estimate = np.array(self.dataset[Constants.CARRIER_MAX_ESTIMATE])
        self.item_zip = self.dataset[Constants.ITEM_ZIP]
        self.buyer_zip = self.dataset[Constants.BUYER_ZIP]
        self.category_id = np.array(self.dataset[Constants.CATEGORY_ID])
        self.item_price = np.array(self.dataset[Constants.ITEM_PRICE])
        self.quantity = np.array(self.dataset[Constants.QUANTITY])
        self.payment_datetime = np.array(self.dataset[Constants.PAYMENT_DATETIME])
        self.delivery_date = np.array(self.dataset[Constants.DELIVERY_DATE])  # LABEL!
        self.weight = np.array(self.dataset[Constants.WEIGHT])
        self.weight_units = np.array(self.dataset[Constants.WEIGHT_UNITS])
        self.package_size = np.array(self.dataset[Constants.PACKAGE_SIZE])

        self.b2c_c2c_to_binary(self.b2c_c2c)
        self.b2c_c2c = np.array(self.b2c_c2c, dtype=int)

        handling_days, shipping_days, delivery_days = self.calculate_handling_and_delivery_days(
            self.acceptance_scan_timestamp,
            self.payment_datetime,
            self.delivery_date)
        # zips = self.add_zip_distance_column(self.item_zip, self.buyer_zip)
        self.convert_weights()
        self.fill_missing_weights()
        self.string_to_numeric_package_size()
        self.fill_missing_package_sizes()
        self.fill_missing_carrier_estimates()
        self.fill_missing_declared_handling_days()
        self.feature_names = [Constants.B2C_C2C, Constants.SELLER_ID, Constants.DECLARED_HANDLING_DAYS,
                              Constants.SHIPMENT_METHOD_ID, Constants.SHIPPING_FEE,
                              Constants.CARRIER_MIN_ESTIMATE, Constants.CARRIER_MAX_ESTIMATE, Constants.CATEGORY_ID,
                              Constants.ITEM_PRICE, Constants.WEIGHT, Constants.PACKAGE_SIZE, Constants.HANDLING_DAYS]
        self.features = np.column_stack((self.b2c_c2c, self.seller_id, self.declared_handling_days,
                                         self.shipment_method_id, self.shipping_fee,
                                         self.carrier_min_estimate, self.carrier_max_estimate, self.category_id,
                                         self.item_price, self.weight, self.package_size, handling_days))
        self.labels = np.array(delivery_days)
        # df = pd.DataFrame(self.features, columns=self.feature_names)
        # print("Features: ")
        # print(df)
        # print("Labels: " + str(list(self.labels)))
        handling_labels = np.array(handling_days)
        shipping_labels = np.array(shipping_days)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features,
                                                                                self.labels,
                                                                                test_size=0.2)
        self.handling_x_train, self.handling_x_test, self.handling_y_train, self.handling_y_test = train_test_split(
            self.features,
            handling_labels,
            test_size=0.2)
        self.shipping_x_train, self.shipping_x_test, self.shipping_y_train, self.shipping_y_test = train_test_split(
            self.features,
            shipping_labels,
            test_size=0.2)
        self.x_train = scale(self.x_train, with_mean=True, with_std=True)
        self.x_test = scale(self.x_test, with_mean=True, with_std=True)
        self.handling_x_train = scale(self.handling_x_train, with_mean=True, with_std=True)
        self.handling_x_test = scale(self.handling_x_test, with_mean=True, with_std=True)
        self.shipping_x_train = scale(self.shipping_x_train, with_mean=True, with_std=True)
        self.shipping_x_test = scale(self.shipping_x_test, with_mean=True, with_std=True)

    @staticmethod
    def b2c_c2c_to_binary(arr):
        """
        Encode b2c_c2c as numeric binary [0,1]
        :param arr:
        """
        if arr[0] in [0, 1]:
            print("Array has already been converted to numeric binary!")
        else:
            for i in range(arr.shape[0]):
                if arr[i][0] == "B":
                    arr[i] = 0
                else:
                    arr[i] = 1

    @staticmethod
    def round_datetime_to_date(datetime):
        """
        Convert all times to the timezome of the buyer
        :param datetime:
        :return: days
        """
        days = datetime.days
        hours = datetime.seconds // 3600
        if hours > 12:
            return days + 1
        else:
            return days

    def calculate_handling_and_delivery_days(self, acceptance_timestamps, payment_timestamps, delivery_date):
        """
        Create labels for handling time (acceptance_scan_timestamp - payment_datetime),
        shipment time (delivery_date - acceptance_scan_timestamp),
        and total time (delivery_date - payment_datetime)
        :param acceptance_timestamps:
        :param payment_timestamps:
        :param delivery_date:
        :return: Tuple of the handling time, shipment time, and total time labels
        """
        handling_labels = []
        shipping_labels = []
        delivery_labels = []
        for i in range(acceptance_timestamps.shape[0]):
            raw_payment = payment_timestamps[i]
            # parse raw_payment time string to separate year, month, date, and time
            p_year, p_month, p_date = int(raw_payment[0:4]), int(raw_payment[5:7]), int(raw_payment[8:10])
            p_hour, p_min, p_sec = int(raw_payment[11:13]), int(raw_payment[14:16]), int(raw_payment[17:19])
            p_datetime = dt.datetime(year=p_year, month=p_month, day=p_date, hour=p_hour, minute=p_min, second=p_sec)

            # parse raw_acceptance time string to separate year, month, date, and time
            raw_acceptance = acceptance_timestamps[i]
            a_year, a_month, a_date = int(raw_acceptance[0:4]), int(raw_acceptance[5:7]), int(raw_acceptance[8:10])
            a_hour, a_min, a_sec = int(raw_acceptance[11:13]), int(raw_acceptance[14:16]), int(raw_acceptance[17:19])
            a_datetime = dt.datetime(year=a_year, month=a_month, day=a_date, hour=a_hour, minute=a_min, second=a_sec)

            raw_delivery = delivery_date[i]
            d_year, d_month, d_date = int(raw_delivery[0:4]), int(raw_delivery[5:7]), int(raw_delivery[8:10])
            d_date = dt.datetime(year=d_year, month=d_month, day=d_date, hour=17)

            # handling days = acceptance time - payment time; shipping days = delivery date - acceptance time
            handling_days = a_datetime - p_datetime
            shipping_days = d_date - a_datetime
            delivery_days = d_date - p_datetime

            # round to nearest day
            rounded_handling_days = self.round_datetime_to_date(handling_days)
            rounded_shipping_days = self.round_datetime_to_date(shipping_days)
            rounded_delivery_days = self.round_datetime_to_date(delivery_days)

            handling_labels.append(rounded_handling_days)
            shipping_labels.append(rounded_shipping_days)
            delivery_labels.append(rounded_delivery_days)

        return np.array(handling_labels), np.array(shipping_labels), np.array(delivery_labels)

    @staticmethod
    def get_distance(item_zip, buyer_zip):
        """
        Quantify distance between buyer and seller using zipcode

        Haversine formula using 'mpu' library which determines the
        great-circle distance between two points on a sphere.
        """
        if item_zip is not None and buyer_zip is not None:
            search = SearchEngine()

            zip1 = search.by_zipcode(item_zip)
            zip2 = search.by_zipcode(buyer_zip)

            if zip1 is None or zip2 is None:
                return None

            lat1 = zip1.lat
            long1 = zip1.lng
            # print(str(lat1) + ", " + str(long1))

            lat2 = zip2.lat
            long2 = zip2.lng
            # print(str(lat2) + ", " + str(long2))

            if lat1 is None or lat2 is None or long1 is None or long2 is None:
                return None

            return mpu.haversine_distance((lat1, long1), (lat2, long2))
        else:
            return None

    def add_zip_distance_column(self, item_zip, buyer_zip):
        """
        Adds a new distance column that calculates
        the zipcode distance between buyer and seller
        :param item_zip:
        :param buyer_zip:
        :return:
        """
        item_zip_str = item_zip.apply(lambda x: str(x))
        buyer_zip_str = buyer_zip.apply(lambda x: str(x))

        zips = pd.concat([item_zip_str, buyer_zip_str], axis=1)

        zips['distance'] = zips.apply(lambda x: self.get_distance(x.item_zip, x.buyer_zip), axis=1)

        return zips['distance']

    def convert_weights(self):
        """
        Use weight_units to convert all weights to the same unit
        """
        for i, unit in enumerate(self.weight_units):
            if unit == 2:
                # convert weight to lbs; 1 kg = 2.20462 lbs.
                self.weight[i] *= 2.20462

    def determine_weight_averages_by_category_id(self):
        """
        Determines weight averages per each category ID
        :return: dictionary object of category_id_weight_means
        """
        category_id_weights = {}
        for i, w in enumerate(self.weight):
            category = self.category_id[i]
            if category not in category_id_weights:
                category_id_weights[category] = [w]
            else:
                category_id_weights[category].append(w)

        category_id_weight_means = {}
        for category in category_id_weights:
            weights = category_id_weights[category]
            average_weight = np.mean(weights)
            category_id_weight_means[category] = average_weight

        return category_id_weight_means

    def fill_missing_weights(self):
        """
        Replace missing weight values with the average weight by category ID
        """
        weight_means = self.determine_weight_averages_by_category_id()
        overall_mean = np.mean(self.weight)
        for i, w in enumerate(self.weight):
            if w == 0:
                # weight is missing, replace with average weight across same category id
                category = self.category_id[i]
                if category in weight_means:
                    self.weight[i] = weight_means[category]
                else:
                    # don't have records for this category id, so replace with overall average
                    self.weight[i] = overall_mean

    def string_to_numeric_package_size(self):
        """
        Encode package_size as discrete numeric values
        """
        if type(self.package_size[0]) == int:
            print("Already converted to discrete numeric values")
        else:
            encodings = {"LETTER": 0, "PACKAGE_THICK_ENVELOPE": 1, "LARGE_ENVELOPE": 2, "VERY_LARGE_PACKAGE": 3,
                         "LARGE_PACKAGE": 4, "EXTRA_LARGE_PACKAGE": 5, "NONE": -1}
            for i, size in enumerate(self.package_size):
                self.package_size[i] = encodings[size]

    def determine_average_weight_by_package_size(self):
        package_size_weights = {}
        for i, w in enumerate(self.weight):
            p_size = self.package_size[i]
            if p_size not in package_size_weights:
                package_size_weights[p_size] = [w]
            else:
                package_size_weights[p_size].append(w)

        package_id_weight_means = {}
        for p_size in package_size_weights:
            weights = package_size_weights[p_size]
            average_weight = np.mean(weights)
            package_id_weight_means[p_size] = average_weight

        return package_id_weight_means

    def fill_missing_package_sizes(self):
        """
        Replace missing package_size values with the most likely size using weight
        """
        weight_means = self.determine_average_weight_by_package_size()
        weight_means.pop(-1, None)
        weight_means_list = [weight_means[key] for key in weight_means]
        for i, s in enumerate(self.package_size):
            if s == -1:
                w = self.weight[i]
                abs_function = lambda value: abs(value - w)
                closest_value = min(weight_means_list, key=abs_function)
                closest_p_size = weight_means_list.index(closest_value)
                self.package_size[i] = closest_p_size

    def determine_average_shipping_estimates_by_shipment_method(self):
        carrier_min_by_shipment_method = {}
        carrier_max_by_shipment_method = {}
        for i, method_id in enumerate(self.shipment_method_id):
            carrier_min = self.carrier_min_estimate[i]
            carrier_max = self.carrier_max_estimate[i]
            if method_id not in carrier_min_by_shipment_method:
                carrier_min_by_shipment_method[method_id] = [carrier_min]
            else:
                carrier_min_by_shipment_method[method_id].append(carrier_min)

            if method_id not in carrier_max_by_shipment_method:
                carrier_max_by_shipment_method[method_id] = [carrier_max]
            else:
                carrier_max_by_shipment_method[method_id].append(carrier_max)

        carrier_min_means = {}
        for method_id in carrier_min_by_shipment_method:
            min_estimates = carrier_min_by_shipment_method[method_id]
            mean_min_estimate = np.mean(min_estimates)
            carrier_min_means[method_id] = mean_min_estimate

        carrier_max_means = {}
        for method_id in carrier_max_by_shipment_method:
            max_estimates = carrier_max_by_shipment_method[method_id]
            mean_max_estimate = np.mean(max_estimates)
            carrier_max_means[method_id] = mean_max_estimate

        return carrier_min_means, carrier_max_means

    def fill_missing_carrier_estimates(self):
        """
        Replace missing carrier_min_estimate and carrier_max_estimate
        with averages from the same shipment_method_id.

        If there are no records for that shipment_method_id, replace
        with overall average across all shipment_method_id's.
        """
        carrier_min_means, carrier_max_means = self.determine_average_shipping_estimates_by_shipment_method()
        overall_min_mean, overall_max_mean = np.mean(self.carrier_min_estimate), np.mean(self.carrier_max_estimate)
        for i, estimate in enumerate(self.carrier_min_estimate):
            if estimate < 0:
                method_id = self.shipment_method_id[i]
                if method_id in carrier_min_means:
                    self.carrier_min_estimate[i] = carrier_min_means[method_id]
                else:
                    self.carrier_min_estimate[i] = overall_min_mean
        for i, estimate in enumerate(self.carrier_max_estimate):
            if estimate < 0:
                method_id = self.shipment_method_id[i]
                if method_id in carrier_max_means:
                    self.carrier_max_estimate[i] = carrier_max_means[method_id]
                else:
                    self.carrier_max_estimate[i] = overall_max_mean

    def fill_missing_declared_handling_days(self):
        """
        Fills in any missing handling days with the mean of all declared handling days
        :return:
        """
        overall_mean = np.mean(self.declared_handling_days)
        for i, days in enumerate(self.declared_handling_days):
            if np.isnan(days):
                self.declared_handling_days[i] = overall_mean
