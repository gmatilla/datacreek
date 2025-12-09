// Gauge panel for sheaf-hypergraph spectral coherence
{
  annotations: { list: [] },
  panels: [
    {
      type: 'gauge',
      title: 'Sheaf-Hypergraph Coherence S',
      datasource: 'Prometheus',
      targets: [{ expr: 'sheaf_hyper_coherence', legendFormat: 'S' }],
    },
  ],
  schemaVersion: 36,
  refresh: '5s',
  title: 'Sheaf-Hypergraph Coherence',
}
