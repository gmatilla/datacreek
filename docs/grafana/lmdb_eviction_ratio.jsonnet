// Converted from JSON to Jsonnet for version control

{
  annotations: { list: [] },
  panels: [
    {
      type: 'graph',
      title: 'LMDB Eviction Ratio TTL/Quota',
      datasource: 'Prometheus',
      targets: [
        { expr: 'rate(lmdb_evictions_total{cause="ttl"}[5m])', legendFormat: 'ttl' },
        { expr: 'rate(lmdb_evictions_total{cause="quota"}[5m])', legendFormat: 'quota' },
      ],
      lines: true,
    },
  ],
  schemaVersion: 36,
  refresh: '5s',
  title: 'LMDB Evictions',
}
