import os
import sqlalchemy
import logging
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, MetaData, Float
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

conn_type = "mysql+pymysql"
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
db_name = os.getenv("DATABASE_NAME")

engine_string = f"{conn_type}://{user}:{password}@{host}:{port}/{db_name}"
# print(engine_string)  # Enable this line if you need to debug, but use caution as it prints your password

engine = sqlalchemy.create_engine(engine_string)

# Set up logging config
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__file__)

# Create a db session
Session = sessionmaker(bind=engine)

class Parameters(Base):
    """Create a data model for the database to be set up for capturing parameters
        Note: this contains the coefficients and intercept for logistic regression
    """

    __tablename__ = "parameters"

    id = Column(Integer, primary_key=True)
    intercept = Column(Float, unique=False, nullable=False)
    trans_price = Column(Float, unique=False, nullable=False)
    owner_dependent = Column(Float, unique=False, nullable=False)
    owner_joint = Column(Float, unique=False, nullable=False)
    owner_self = Column(Float, unique=False, nullable=False)
    ticker_AMZN = Column(Float, unique=False, nullable=False)
    ticker_FB = Column(Float, unique=False, nullable=False)
    ticker_MSFT = Column(Float, unique=False, nullable=False)
    ticker_NTAP = Column(Float, unique=False, nullable=False)
    ticker_NVDA = Column(Float, unique=False, nullable=False)
    ticker_RUN = Column(Float, unique=False, nullable=False)
    ticker_TSLA = Column(Float, unique=False, nullable=False)
    type_sale_full = Column(Float, unique=False, nullable=False)
    type_sale_partial = Column(Float, unique=False, nullable=False)
    amount_1 = Column(Float, unique=False, nullable=False)
    amount_2 = Column(Float, unique=False, nullable=False)
    amount_3 = Column(Float, unique=False, nullable=False)
    amount_4 = Column(Float, unique=False, nullable=False)
    amount_5 = Column(Float, unique=False, nullable=False)
    amount_6 = Column(Float, unique=False, nullable=False)
    amount_7 = Column(Float, unique=False, nullable=False)
    amount_8 = Column(Float, unique=False, nullable=False)
    Dean_Phillips = Column(Float, unique=False, nullable=False)
    Donald_Sternoff_Beyer = Column(Float, unique=False, nullable=False)
    Gilbert_Cisneros = Column(Float, unique=False, nullable=False)
    Josh_Gottheimer = Column(Float, unique=False, nullable=False)
    Kevin_Hern = Column(Float, unique=False, nullable=False)
    Kurt_Schrader = Column(Float, unique=False, nullable=False)
    Michael_T_McCaul = Column(Float, unique=False, nullable=False)
    Nancy_Pelosi = Column(Float, unique=False, nullable=False)
    Rohit_Khanna = Column(Float, unique=False, nullable=False)

    def __repr__(self):
        return f"<Parameter {self.id}>"

class Input(Base):
    """Create a data model for the database to be set up for capturing user input
        Note: this contains the selections made by the user while interacting with the live app
    """

    __tablename__ = "user_inputs"

    id = Column(Integer, primary_key=True)
    trans_price = Column(Float, unique=False, nullable=False)
    owner_type = Column(String(100), unique=False, nullable=False)
    ticker= Column(String(100), unique=False, nullable=False)
    transaction_type = Column(String(100), unique=False, nullable=False)
    amount = Column(String(100), unique=False, nullable=False)
    representative = Column(String(100), unique=False, nullable=False)

    def __repr__(self):
        return f"<Input {self.id}>"

if __name__=="__main__":
    engine = sqlalchemy.create_engine(engine_string)
    Base.metadata.create_all(engine)
