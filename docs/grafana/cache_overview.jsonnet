// Converted from JSON to Jsonnet for version control

{
  annotations: { list: [] },
  panels: [
    {
      type: 'graph',
      title: 'LMDB Evictions by Cause',
      datasource: 'Prometheus',
      targets: [
        { expr: 'rate(lmdb_evictions_total{cause="ttl"}[5m])', legendFormat: 'ttl' },
        { expr: 'rate(lmdb_evictions_total{cause="quota"}[5m])', legendFormat: 'quota' },
        { expr: 'rate(lmdb_evictions_total{cause="manual"}[5m])', legendFormat: 'manual' },
      ],
      lines: true,
    },
    {
      type: 'graph',
      title: 'Redis Hit Ratio',
      datasource: 'Prometheus',
      targets: [{ expr: 'redis_hit_ratio', legendFormat: 'hit' }],
      lines: true,
    },
    {
      type: 'graph',
      title: 'Ingest Queue Fill Ratio',
      datasource: 'Prometheus',
      targets: [{ expr: 'ingest_queue_fill_ratio', legendFormat: 'fill' }],
      lines: true,
    },
  ],
  schemaVersion: 36,
  refresh: '5s',
  title: 'Cache Overview',
}
