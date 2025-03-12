import akshare as ak
import pandas as pd
from app.utils.logger import get_logger
import concurrent.futures

logger = get_logger()


class StockSearchService:
    """Service for searching stocks by code or name"""

    def __init__(self):
        # Cache for stock lists (to avoid repeated API calls)
        self._a_stock_list = None
        self._hk_stock_list = None

    def _get_a_stock_list(self):
        """Get A-share stock list from akshare"""
        if self._a_stock_list is None:
            try:
                logger.info("Fetching A-share stock list")
                # Get A-share stock information
                self._a_stock_list = ak.stock_info_a_code_name()
                self._a_stock_list = self._a_stock_list.rename(
                    columns={"code": "stock_code", "name": "stock_name"}
                )
                # Add market type
                self._a_stock_list["market_type"] = "A"
                logger.info(f"Fetched {len(self._a_stock_list)} A-share stocks")
            except Exception as e:
                logger.error(f"Error fetching A-share stock list: {str(e)}")
                logger.exception(e)
                self._a_stock_list = pd.DataFrame(
                    columns=["stock_code", "stock_name", "market_type"]
                )

        return self._a_stock_list

    def _get_hk_stock_list(self):
        """Get Hong Kong stock list from akshare"""
        if self._hk_stock_list is None:
            try:
                logger.info("Fetching Hong Kong stock list")
                # Get HK stock information
                self._hk_stock_list = ak.stock_hk_spot_em()

                # Rename columns to match A-share format
                self._hk_stock_list = self._hk_stock_list[["代码", "名称"]].rename(
                    columns={"代码": "stock_code", "名称": "stock_name"}
                )

                # Add market type
                self._hk_stock_list["market_type"] = "HK"
                logger.info(f"Fetched {len(self._hk_stock_list)} Hong Kong stocks")
            except Exception as e:
                logger.error(f"Error fetching Hong Kong stock list: {str(e)}")
                logger.exception(e)
                self._hk_stock_list = pd.DataFrame(
                    columns=["stock_code", "stock_name", "market_type"]
                )

        return self._hk_stock_list

    def _search_market(self, stocks_df, keyword):
        """Helper method to search within a specific market dataframe"""
        return stocks_df[
            (stocks_df["stock_code"].str.contains(keyword))
            | (stocks_df["stock_name"].str.lower().str.contains(keyword))
        ]

    def search_stocks(self, keyword, max_results=20):
        """
        Search stocks by code or name in both A and HK markets in parallel

        Args:
            keyword (str): Search keyword
            max_results (int): Maximum number of results to return

        Returns:
            list: List of matching stocks from both markets
        """
        if not keyword:
            return []

        keyword = keyword.strip().lower()
        logger.info(f"Searching for stocks with keyword: '{keyword}' in all markets")

        try:
            # Use ThreadPoolExecutor to fetch and search both markets in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks for both markets
                a_future = executor.submit(
                    self._search_market, self._get_a_stock_list(), keyword
                )
                hk_future = executor.submit(
                    self._search_market, self._get_hk_stock_list(), keyword
                )

                # Get results from both markets
                a_results = a_future.result()
                hk_results = hk_future.result()

                # Combine results from both markets
                combined_results = (
                    pd.concat([a_results, hk_results])
                    if not (a_results.empty and hk_results.empty)
                    else pd.DataFrame()
                )

                # Sort by code for consistent results
                if not combined_results.empty:
                    combined_results = combined_results.sort_values(
                        "stock_code"
                    ).reset_index(drop=True)

                    # Limit results
                    if len(combined_results) > max_results:
                        combined_results = combined_results.head(max_results)

                    logger.info(
                        f"Found {len(combined_results)} stocks matching '{keyword}'"
                    )

                    # Convert to list of dicts for JSON response
                    return combined_results.to_dict("records")

                return []

        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            logger.exception(e)
            return []
