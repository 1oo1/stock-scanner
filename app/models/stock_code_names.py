from app import db
from datetime import datetime


class StockCodeName(db.Model):
    """Model for storing stock codes and names for different markets"""

    __tablename__ = "stock_code_names"

    id = db.Column(db.Integer, primary_key=True)
    stock_code = db.Column(db.String(20), nullable=False, index=True)
    stock_name = db.Column(db.String(100), nullable=False, index=True)
    market_type = db.Column(db.String(10), nullable=False)  # A, HK, etc.
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        db.UniqueConstraint("stock_code", "market_type", name="uix_stock_code_market"),
    )

    @classmethod
    def bulk_upsert(cls, stocks_df):
        """
        Bulk update or insert stock data from a DataFrame

        Args:
            stocks_df (DataFrame): DataFrame with stock_code, stock_name, and market_type columns
        """
        if stocks_df.empty:
            return

        try:
            # Get all existing records matching the stock codes in the DataFrame in a single query
            stock_codes = stocks_df["stock_code"].unique().tolist()
            existing_stocks = {
                (s.stock_code, s.market_type): s
                for s in cls.query.filter(cls.stock_code.in_(stock_codes)).all()
            }

            new_records = []
            for _, row in stocks_df.iterrows():
                # Check if record already exists using both stock_code and market_type
                # If found and the name is different, update the name
                key = (row["stock_code"], row["market_type"])
                if key in existing_stocks:
                    # Update existing record
                    existing = existing_stocks[key]
                    if existing.stock_name != row["stock_name"]:
                        existing.stock_name = row["stock_name"]
                        existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    new_records.append(
                        cls(
                            stock_code=row["stock_code"],
                            stock_name=row["stock_name"],
                            market_type=row["market_type"],
                        )
                    )

            # Add all new records at once
            if new_records:
                db.session.bulk_save_objects(new_records)

            # Commit all changes (both updates and inserts)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    @classmethod
    def search(cls, keyword, max_results=20):
        """
        Search stocks by code or name

        Args:
            keyword (str): Search keyword
            max_results (int): Maximum number of results to return

        Returns:
            list: List of dictionaries with stock_code, stock_name, and market_type
        """
        keyword = f"%{keyword.lower()}%"
        results = (
            cls.query.filter(
                db.or_(
                    db.func.lower(cls.stock_code).like(keyword),
                    db.func.lower(db.func.replace(cls.stock_name, " ", "")).like(
                        keyword.replace(" ", "")
                    ),
                )
            )
            .order_by(cls.stock_code)
            .limit(max_results)
            .all()
        )

        return [
            {
                "stock_code": item.stock_code,
                "stock_name": item.stock_name,
                "market_type": item.market_type,
            }
            for item in results
        ]

    @classmethod
    def get_by_market(cls, market_type):
        """
        Get all stocks for a specific market

        Args:
            market_type (str): Market type (A, HK, etc.)

        Returns:
            list: List of dictionaries with stock_code and stock_name
        """
        results = cls.query.filter_by(market_type=market_type).all()
        return [
            {
                "stock_code": item.stock_code,
                "stock_name": item.stock_name,
                "market_type": item.market_type,
            }
            for item in results
        ]

    @classmethod
    def should_update(cls):
        """
        When the table is empty, we need to update it.
        When the update time is older than 48 hours, we need to update it.
        """
        # Check if the table is empty
        if cls.query.count() == 0:
            return True

        # Check if the last update was more than 48 hours ago
        last_update = cls.query.order_by(cls.updated_at.desc()).first()
        if (
            last_update
            and (datetime.utcnow() - last_update.updated_at).total_seconds() > 48 * 3600
        ):
            return True

        return False
