from ..utils.excel_options import ExcelOperations


class ExcelRW:
    excel = ExcelOperations()

    def save_excel(self, sheet_name, data, filename, header_data):
        self.excel.set_data(sheet_name, data, filename, header_data)
