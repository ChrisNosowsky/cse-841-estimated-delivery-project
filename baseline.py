# ==============================================================================
# Baseline Data
#
# Authors: Christopher Nosowsky
#
# ==============================================================================

import pandas as pd
import numpy as np
import postprocess
from datetime import datetime


class Baseline:
    """
    Baseline is our target goal.
    Try to have model performance that exceeds this baseline.
    """

    EST_AVAIL_DV_DT = "EST_AVAIL_DV_DT"
    DELIVERY_SCAN_TMSTP = "DELIVERY_SCAN_TMSTP"
    CLASSIFICATION_DESC = "CLASSIFICATION_DESC"
    INMARKET_ROUTE_LEGS_DESC = "INMARKET_ROUTE_LEGS_DESC"
    SHIP_DATE = "SHIP_DATE"
    RESIDENTIAL = "RESIDENTIAL"
    BUSINESS = "BUSINESS"
    UNKNOWN = "UNKNOWN"
    EST_TIME = "estTime"
    MONTH_ENCODING = {"01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
                      "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
                      "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC"}

    def __init__(self, filename):
        self.filename = filename
        self.baseline_data = pd.read_csv(filename, engine='python')
        if postprocess.is_debug():
            print("Debug Mode Activated")
            self.baseline_data = self.baseline_data.head(5000)

        self.backup_est_dv_dt = np.array(self.baseline_data[Baseline.EST_AVAIL_DV_DT])

        self.actual_delivery_date = np.array(self.baseline_data[Baseline.DELIVERY_SCAN_TMSTP])
        self.remove_timestamp_from_actual_delivery_date()

        self.classification = np.array(self.baseline_data[Baseline.CLASSIFICATION_DESC])
        self.inmarket_data = np.array(self.baseline_data[Baseline.INMARKET_ROUTE_LEGS_DESC])
        self.ship_date = np.array(self.baseline_data[Baseline.SHIP_DATE])

        self.estimated_delivery_date = self.extract_est_delivery_date_from_seller_loc()

        # Convert to number of days it took to ship package
        self.actual_delivery_days = self.get_baseline_shipment_delivery_days(self.actual_delivery_date)
        self.estimated_delivery_days = self.get_baseline_shipment_delivery_days(self.estimated_delivery_date)

    def remove_timestamp_from_actual_delivery_date(self):
        for i in range(self.actual_delivery_date.shape[0]):
            self.actual_delivery_date[i] = self.actual_delivery_date[i][0:9]

    def get_baseline_target_percentage(self):
        """
        Gets baseline percentage.
        Takes estimated delivery and actual delivery and scores the right versus wrong predictions
        with the current system
        :return:
        """
        count = 0
        total = 0
        for i, d in enumerate(self.estimated_delivery_date):
            if d is not None and self.actual_delivery_date[i] is not None:
                total += 1
                if self.actual_delivery_date[i] == d:
                    count += 1

        print("Total Records: " + str(total))
        print("Correct Estimates: " + str(count))
        return 100 * (count / total)

    def get_residential_target_percentage(self):
        count = 0
        total = 0
        for i, d in enumerate(self.estimated_delivery_date):
            if d is not None and self.actual_delivery_date[i] is not None \
                    and self.classification[i] == Baseline.RESIDENTIAL:
                total += 1
                if self.actual_delivery_date[i] == d:
                    count += 1

        print("Total Residential Records: " + str(total))
        print("Correct Residential Estimates: " + str(count))
        return 100 * (count / total)

    def get_business_target_percentage(self):
        count = 0
        total = 0
        for i, d in enumerate(self.estimated_delivery_date):
            if d is not None and self.actual_delivery_date[i] is not None \
                    and self.classification[i] == Baseline.BUSINESS:
                total += 1
                if self.actual_delivery_date[i] == d:
                    count += 1

        print("Total Business Records: " + str(total))
        print("Correct Business Estimates: " + str(count))
        return 100 * (count / total)

    def get_unknown_target_percentage(self):
        count = 0
        total = 0
        for i, d in enumerate(self.estimated_delivery_date):
            if d is not None and self.actual_delivery_date[i] is not None \
                    and self.classification[i] == Baseline.UNKNOWN:
                total += 1
                if self.actual_delivery_date[i] == d:
                    count += 1

        print("Total Unknown Records: " + str(total))
        print("Correct Unknown Estimates: " + str(count))
        return 100 * (count / total)

    def get_target_loss(self):
        return postprocess.evaluate_loss(self.estimated_delivery_days,
                                         self.actual_delivery_days)

    @staticmethod
    def get_month_abbrev(month):
        return Baseline.MONTH_ENCODING.get(month)

    def extract_est_delivery_date_from_seller_loc(self):
        est_avail_dv_dt = []
        for i in range(self.inmarket_data.shape[0]):
            lst_obj = list(eval(self.inmarket_data[i]))
            est_time = lst_obj[0].get(Baseline.EST_TIME)
            est_yr, est_mo, est_day = est_time[2:4], self.get_month_abbrev(est_time[4:6]), est_time[6:8]
            if est_time == '' or est_time is None or len(est_time) == 0:
                est_time = self.backup_est_dv_dt[i]
            else:
                est_time = str(est_day + '-' + est_mo + '-' + est_yr)
            est_avail_dv_dt.append(est_time)
        return np.array(est_avail_dv_dt)

    @staticmethod
    def round_datetime_to_date(date_time):
        """
        Convert all times to the timezome of the buyer
        :param date_time:
        :return: days
        """
        days = date_time.days
        hours = date_time.seconds // 3600
        if hours > 12:
            return days + 1
        else:
            return days

    def get_baseline_shipment_delivery_days(self, delivery_date):
        delivery_labels = []
        for i in range(self.actual_delivery_date.shape[0]):
            ship_dt = self.ship_date[i]
            delivery_dt = delivery_date[i]

            ship_datetime_object = datetime.strptime(ship_dt, '%d-%b-%y')
            delivery_datetime_object = datetime.strptime(delivery_dt, '%d-%b-%y')
            delivery_days = delivery_datetime_object - ship_datetime_object
            rounded_delivery_days = self.round_datetime_to_date(delivery_days)
            delivery_labels.append(rounded_delivery_days)

        return np.array(delivery_labels)
