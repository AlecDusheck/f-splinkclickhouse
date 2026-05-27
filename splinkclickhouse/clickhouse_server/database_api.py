import pandas as pd
from clickhouse_connect.driver.client import Client

from ..database_api import ClickhouseAPI
from .dataframe import ClickhouseServerDataFrame


class ClickhouseServerAPI(ClickhouseAPI):
    def __init__(
        self,
        client: Client,
        register_custom_udfs: bool = True,
        *,
        max_parallel_replicas: int = 1,
    ):
        super().__init__()

        self.client = client
        self._max_parallel_replicas = max_parallel_replicas
        self._pending_insert: tuple[str, str] | None = None
        self.set_union_default_mode()
        if max_parallel_replicas > 1:
            self.set_parallel_replicas(max_parallel_replicas)
        self._create_random_function()
        if register_custom_udfs:
            self._register_custom_udfs()

    def _table_registration(self, input, table_name) -> None:
        input = self._coerce_input_to_pd_if_needed(input)

        if isinstance(input, pd.DataFrame):
            sql = self._create_table_sql_from_pd_frame(input, table_name)
            self._execute_sql_against_backend(sql)
            self.client.insert_df(table_name, input)
        else:
            raise TypeError(
                "ClickhouseServerAPI currently only accepts table names (str) "
                "or pandas DataFrames as inputs for table registration. "
                f"Received type {type(input)}"
            )

    def table_to_splink_dataframe(self, templated_name, physical_name):
        return ClickhouseServerDataFrame(templated_name, physical_name, self)

    def table_exists_in_database(self, table_name):
        sql = self._information_schema_query(
            "table_name", "tables", table_name, self.database
        )

        res = self.client.query(sql).result_set
        return len(res) > 0

    def _create_table_as(self, physical_name: str, sql: str) -> str:
        # CTAS-of-a-parallel-SELECT is miscompiled on ClickHouse 25.12
        # (THERE_IS_NO_COLUMN / UNKNOWN_TABLE — see tests_local/test_parallel_replicas).
        # So under parallel replicas, create the structure with zero rows, then
        # populate via INSERT...SELECT (queued here, run in _execute_sql_against_backend),
        # whose SELECT *does* fan out across replicas.
        if self._max_parallel_replicas > 1:
            self._pending_insert = (physical_name, sql)
            return f"CREATE TABLE {physical_name} ORDER BY tuple() AS SELECT * FROM ({sql}) LIMIT 0"
        return super()._create_table_as(physical_name, sql)

    def _execute_sql_against_backend(
        self, final_sql: str, templated_name: str = None, physical_name: str = None
    ):
        pending = self._pending_insert
        self._pending_insert = None
        if pending is None:
            self.client.query(final_sql)
            return
        # Structure-create reads no rows -> run it single-replica (sidesteps the
        # CTAS bug); the INSERT's SELECT then parallelises across replicas.
        physical_name, inner_sql = pending
        self.client.query(final_sql, settings={"enable_parallel_replicas": 0})
        self.client.query(f"INSERT INTO {physical_name} SELECT * FROM ({inner_sql})")

    def _get_results_from_backend(self, sql: str):
        res = self.client.query(sql).named_results()
        return res

    @property
    def database(self) -> str:
        return self.client.database or "default"

    # Clickhouse can not handle a bare 'UNION' by default
    # we can set desired behaviour for the session by executing this
    def set_union_default_mode(self) -> None:
        """
        Set `union_default_mode` to 'DISTINCT'

        Splink uses SQL involving bare 'UNION's in order to combine rows _without_
        removing duplicates.
        Clickhouse requires this to be explicitly set, so this function changes the
        server setting so that this is the default behaviour.
        See [the Clickhouse docs](https://clickhouse.com/docs/en/sql-reference/statements/select/union).

        This happens automatically on creation of `ClickhouseServerAPI`.
        However, this setting is tied to a session, so if this expires then this
        configuration value will be forgotten, and you may receive errors such as
        `Expected ALL or DISTINCT in SelectWithUnion query`.
        In this case you will need to manually call this function.
        """  # noqa: E501 (line too long)
        self._execute_sql_against_backend("SET union_default_mode = 'DISTINCT'")

    def set_parallel_replicas(self, max_parallel_replicas: int) -> None:
        """Spread every query across up to ``max_parallel_replicas`` ClickHouse
        compute replicas (ClickHouse Cloud, or a self-managed cluster).
        Session-scoped, like ``set_union_default_mode``.

        The per-pair scoring workload (the bulk of the work) is SELECT-heavy and
        scales near-linearly across replicas. Splink materialises every step via
        ``CREATE TABLE ... AS``, and CTAS-of-a-parallel-SELECT is miscompiled on
        ClickHouse 25.12; ``_create_table_as`` works around this by splitting it
        into a zero-row structure-create plus a parallel ``INSERT ... SELECT``.
        """
        for setting, value in (
            ("enable_parallel_replicas", 1),
            ("max_parallel_replicas", max_parallel_replicas),
            # ClickHouse Cloud tables are SharedMergeTree; this also covers
            # plain MergeTree on a self-managed cluster.
            ("parallel_replicas_for_non_replicated_merge_tree", 1),
        ):
            self._execute_sql_against_backend(f"SET {setting} = {value}")

    def _create_table_sql_from_pd_frame(self, df: pd.DataFrame, table_name: str) -> str:
        sql = f"CREATE OR REPLACE TABLE {table_name} ("

        first_col = True
        for column_name in df.columns:
            if not first_col:
                sql += ", "
            column = df[column_name]
            col_type = column.dtype
            first_col = False

            if pd.api.types.is_unsigned_integer_dtype(col_type):
                sql += f"{column_name} Nullable(UInt32)"
            elif pd.api.types.is_integer_dtype(col_type):
                sql += f"{column_name} Nullable(Int32)"
            elif pd.api.types.is_float_dtype(col_type):
                sql += f"{column_name} Nullable(Float64)"
            elif pd.api.types.is_list_like(column[0]):
                sql += f"{column_name} Array(String)"
            elif pd.api.types.is_string_dtype(col_type):
                sql += f"{column_name} Nullable(String)"
            elif pd.api.types.is_datetime64_dtype(col_type):
                sql += f"{column_name} Nullable(DateTime64)"
            else:
                raise ValueError(f"Unknown data type {col_type}")

        sql += ") ENGINE MergeTree ORDER BY tuple()"

        return sql
