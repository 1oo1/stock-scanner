import pandas as pd
from app.utils.logger import get_logger
from app.utils.akshare import get_akshare
from app.models.stock_code_names import StockCodeName

logger = get_logger()

is_updating = False


class StockSearchService:
    """Service for searching stocks by code or name"""

    def _update_stock_names(self):
        """Update stock names in the database"""

        global is_updating
        is_updating = True

        try:
            # Get A-share stock list
            a_stock_list = get_akshare().stock_info_a_code_name()
            a_stock_list = a_stock_list.rename(
                columns={"code": "stock_code", "name": "stock_name"}
            )
            a_stock_list["market_type"] = "A"

            # Get HK stock list
            hk_stock_list = get_akshare().stock_hk_spot_em()
            hk_stock_list = hk_stock_list[["代码", "名称"]].rename(
                columns={"代码": "stock_code", "名称": "stock_name"}
            )
            hk_stock_list["market_type"] = "HK"

            # Combine both lists
            combined_stocks = pd.concat([a_stock_list, hk_stock_list])

            # Bulk upsert into database
            StockCodeName.bulk_upsert(combined_stocks)

        except Exception as e:
            logger.error(f"Error updating stock names: {str(e)}")
            logger.exception(e)

        finally:
            is_updating = False

    def search_stocks(self, keyword, max_results=20):
        """
        Search stocks by code or name.

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
            # First try to search directly in database
            db_results = StockCodeName.search(keyword, max_results)

            # Call _update_stock_names asynchronously
            global is_updating
            if not db_results and not is_updating and StockCodeName.should_update():
                # Get the actual app object
                self._update_stock_names()
                db_results = StockCodeName.search(keyword, max_results)

            logger.info(
                f"Found {len(db_results)} stocks matching '{keyword}' in database"
            )

            return db_results

        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            logger.exception(e)
            return []
