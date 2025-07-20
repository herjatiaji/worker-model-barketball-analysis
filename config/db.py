from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import BigInteger, SmallInteger, String, JSON, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import BIGINT, SMALLINT
from config.settings import settings

engine = create_engine(settings.DB_URL)
metadata = MetaData()

# Define tables
videos = Table("videos", metadata,
    Column("id", BIGINT(unsigned=True), primary_key=True, autoincrement=True),
    Column("thumbnail_url", String(255)),
)

video_results = Table("video_results", metadata,
    Column("id", BIGINT(unsigned=True), primary_key=True, autoincrement=True),
    Column("video_id", BIGINT(unsigned=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("court_length_px", SMALLINT(unsigned=True), nullable=False),
    Column("court_width_px", SMALLINT(unsigned=True), nullable=False),
    Column("video_url", String(255), nullable=False),
    Column("tracking", JSON, nullable=False),
    Column("shot", JSON, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
)