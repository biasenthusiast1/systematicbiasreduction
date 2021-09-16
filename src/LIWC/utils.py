"""
16.2.2021
Author: Shirin
"""
import csv


def open_file_to_read(file_path):
    """
    opens a file to read
    :param file_path:
    :return:
    """
    file = open(file_path, "r", encoding="utf-8")
    return file


def open_file_to_write(file_path):
    """
    opens a file to write
    :param file_path:
    :return: file
    """
    file = open(file_path, "w", encoding="utf-8")
    return file


def open_csv(file_path):
    """
    reads a csv file
    :param file_path:
    :return: csv_reader
    """
    csv_file = open(file_path, "r", encoding="utf-8")
    csv_reader = csv.reader(csv_file, delimiter=',')
    return csv_reader
