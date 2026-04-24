import 'package:flutter/material.dart';
import 'mock_data.dart';

class DatasetBuilderPage extends StatefulWidget {
  const DatasetBuilderPage({super.key});

  @override
  State<DatasetBuilderPage> createState() => _DatasetBuilderPageState();
}

class _DatasetBuilderPageState extends State<DatasetBuilderPage> {
  bool _building = false;
  bool _built = false;

  void _buildDataset() async {
    setState(() { _building = true; _built = false; });
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) setState(() { _building = false; _built = true; });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          const Row(
            children: [
              Icon(Icons.dataset_outlined, color: Color(0xFF6366F1), size: 22),
              SizedBox(width: 10),
              Text('Dataset Builder', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B))),
            ],
          ),
          const SizedBox(height: 8),
          const Text('Build a versioned training dataset from approved samples.',
            style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
          ),
          const SizedBox(height: 28),

          // Stats cards row
          Row(
            children: [
              _statCard('Total Approved', '$mockApprovedTotal', Icons.check_circle_outline, const Color(0xFF22C55E)),
              const SizedBox(width: 16),
              _statCard('Images', '$mockApprovedImages', Icons.image_outlined, const Color(0xFF8B5CF6)),
              const SizedBox(width: 16),
              _statCard('Videos', '$mockApprovedVideos', Icons.videocam_outlined, const Color(0xFF3B82F6)),
              const SizedBox(width: 16),
              _statCard('Audio', '$mockApprovedAudios', Icons.mic_outlined, const Color(0xFFF97316)),
            ],
          ),
          const SizedBox(height: 28),

          // Breakdown chart (visual bar)
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Approved Samples Breakdown',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B)),
                ),
                const SizedBox(height: 16),
                _barRow('Images', mockApprovedImages, mockApprovedTotal, const Color(0xFF8B5CF6)),
                const SizedBox(height: 10),
                _barRow('Videos', mockApprovedVideos, mockApprovedTotal, const Color(0xFF3B82F6)),
                const SizedBox(height: 10),
                _barRow('Audio', mockApprovedAudios, mockApprovedTotal, const Color(0xFFF97316)),
              ],
            ),
          ),
          const SizedBox(height: 28),

          // Build action
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.inventory_2_outlined, size: 20, color: Color(0xFF6366F1)),
                    const SizedBox(width: 10),
                    const Text('Dataset Version',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B)),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: const Color(0xFF6366F1).withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(mockDatasetVersion,
                        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: Color(0xFF6366F1), fontFamily: 'monospace'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                const Text(
                  'Build a new dataset snapshot from all approved samples. '
                  'This will package images, videos, and audio files into a versioned dataset ready for model retraining.',
                  style: TextStyle(fontSize: 13, color: Color(0xFF64748B), height: 1.5),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    SizedBox(
                      height: 42,
                      child: ElevatedButton.icon(
                        onPressed: _building ? null : _buildDataset,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF6366F1),
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                          elevation: 0,
                          padding: const EdgeInsets.symmetric(horizontal: 24),
                        ),
                        icon: _building
                            ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                            : const Icon(Icons.build_circle_outlined, size: 18),
                        label: Text(_building ? 'Building...' : 'Build Dataset'),
                      ),
                    ),
                    if (_built) ...[
                      const SizedBox(width: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: const Color(0xFF22C55E).withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.check_circle, size: 16, color: Color(0xFF22C55E)),
                            SizedBox(width: 6),
                            Text('Dataset built successfully!',
                              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF22C55E)),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _statCard(String label, String value, IconData icon, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.15)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 24, color: color),
            const SizedBox(height: 12),
            Text(value, style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: color)),
            const SizedBox(height: 4),
            Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF94A3B8))),
          ],
        ),
      ),
    );
  }

  Widget _barRow(String label, int count, int total, Color color) {
    final pct = total > 0 ? count / total : 0.0;
    return Row(
      children: [
        SizedBox(width: 60, child: Text(label, style: const TextStyle(fontSize: 13, color: Color(0xFF64748B)))),
        const SizedBox(width: 12),
        Expanded(
          child: Stack(
            children: [
              Container(height: 8, decoration: BoxDecoration(color: const Color(0xFFF1F5F9), borderRadius: BorderRadius.circular(4))),
              FractionallySizedBox(
                widthFactor: pct,
                child: Container(height: 8, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(4))),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Text('$count', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: color)),
      ],
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
      ),
      child: child,
    );
  }
}
