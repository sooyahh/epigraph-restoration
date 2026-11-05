"""NRICH field mappings and shared constants."""

NRICH_BASE_URL = "http://portal.nrich.go.kr"

# Verified against KSM_IDX=3,5 (docs/DATA_SCHEMA.md §18.5).
NRICH_MD_DATA_TO_FIELD = {
    "md_data1": "ROW_NUM",
    "md_data2": "KSM_IDX",
    "md_data3": "KSM_KOR_NAME",
    "md_data4": "KSM_CHI_NAME",
    "md_data5": "KSM_EAR",
    "md_data6": "KSM_EAR_NM",
    "md_data7": "KSM_AGE",
    "md_data8": "KSM_TYPE",
    "md_data9": "KSM_TYPENM",
    "md_data10": "KSM_LIFESIZE",
    "md_data11": "KSM_LOCATION_ADDR",
    "md_data12": "KSM_FONT",
    "md_data13": "KSM_PROD_CJ",
    "md_data14": "KSM_PROD_GJ",
    "md_data15": "KSM_PROD_SJ",
    "md_data16": "KSM_USE_SCOPE",
    "md_data17": "FORMAL_CDNM",
    "md_data18": "KSM_SUVEY",
    "md_data19": "KSM_COPYRIGHT",
    "md_data20": "FORMAL_CD",
    "md_data21": "USE_SCOPE_NM",
    "md_data22": "DATA_LINK",
}

DETAIL_VIEW_PATH = "/kor/ksmUsrView.do"
PANDOK_VIEW_PATH = "/kor/ksmPandokView.do"
PANDOK_WRITER_PATH = "/kor/ksmPandokWriter.do"
PANDOK_CLOB_PATH = "/ajaxPandokClob.do"
LIST_VIEW_PATH = "/kor/ksmUsrList.do"

DEFAULT_MENU_IDX = "584"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gsm-capstone/1.0)"}
