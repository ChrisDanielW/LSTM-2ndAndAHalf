from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Define the database file (SQLite)
DATABASE_URL = "sqlite:///stocksage.db"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Base class for declarative models
Base = declarative_base()

# Define the User table
class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(Date, nullable=False)

# Define the Stock table
class Stock(Base):
    __tablename__ = 'stock'
    stock_id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False)
    company_name = Column(String(100))
    sector = Column(String(50))
    industry = Column(String(50))

# Define the StockData table
class StockData(Base):
    __tablename__ = 'stock_data'
    data_id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stock.stock_id'), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)

    # Relationship to Stock table
    stock = relationship("Stock", back_populates="stock_data")

# Define the Prediction table
class Prediction(Base):
    __tablename__ = 'prediction'
    prediction_id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stock.stock_id'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.user_id'), nullable=False)
    prediction_date = Column(Date, nullable=False)
    predicted_price = Column(Float)
    prediction_for_date = Column(Date, nullable=False)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    user = relationship("User", back_populates="predictions")

# Add relationships to Stock and User tables
Stock.stock_data = relationship("StockData", order_by=StockData.data_id, back_populates="stock")
Stock.predictions = relationship("Prediction", order_by=Prediction.prediction_id, back_populates="stock")
User.predictions = relationship("Prediction", order_by=Prediction.prediction_id, back_populates="user")

# Create all tables in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session() 