import 'package:flutter/material.dart';
import 'mock_data.dart';

class RetainedSamplesPage extends StatefulWidget {
  const RetainedSamplesPage({super.key});

  @override
  State<RetainedSamplesPage> createState() => _RetainedSamplesPageState();
}

class _RetainedSamplesPageState extends State<RetainedSamplesPage> {
  late List<MockSample> _samples;
  String _filterStatus = 'all';
  String _filterType = 'all';

  @override
  void initState() {
    super.initState();
    _samples = List.from(mockSamples);
  }

  List<MockSample> get _filteredSamples {
    return _samples.where((s) {
      if (_filterStatus != 'all' && s.status != _filterStatus) return false;
      if (_filterType != 'all' && s.fileType != _filterType) return false;
      return true;
    }).toList();
  }

  void _updateStatus(String id, String newStatus) {
    setState(() {
      final idx = _samples.indexWhere((s) => s.id == id);
      if (idx != -1) {
        final old = _samples[idx];
        _samples[idx] = MockSample(
          id: old.id, fileType: old.fileType, fileName: old.fileName,
          uploadTime: old.uploadTime, groundTruth: old.groundTruth,
          status: newStatus,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final filtered = _filteredSamples;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          children: [
            const Icon(Icons.storage_rounded, color: Color(0xFF6366F1), size: 22),
            const SizedBox(width: 10),
            const Text('Retained Samples',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B)),
            ),
            const Spacer(),
            _badge('${_samples.length} total', const Color(0xFF6366F1)),
            const SizedBox(width: 8),
            _badge('${_samples.where((s) => s.status == 'approved').length} approved', const Color(0xFF22C55E)),
            const SizedBox(width: 8),
            _badge('${_samples.where((s) => s.status == 'pending').length} pending', const Color(0xFFF59E0B)),
          ],
        ),
        const SizedBox(height: 16),

        // Filters
        Row(
          children: [
            const Text('Status: ', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF64748B))),
            _filterChip('All', 'all', _filterStatus, (v) => setState(() => _filterStatus = v)),
            _filterChip('Pending', 'pending', _filterStatus, (v) => setState(() => _filterStatus = v)),
            _filterChip('Approved', 'approved', _filterStatus, (v) => setState(() => _filterStatus = v)),
            _filterChip('Rejected', 'rejected', _filterStatus, (v) => setState(() => _filterStatus = v)),
            const SizedBox(width: 20),
            const Text('Type: ', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF64748B))),
            _filterChip('All', 'all', _filterType, (v) => setState(() => _filterType = v)),
            _filterChip('Image', 'image', _filterType, (v) => setState(() => _filterType = v)),
            _filterChip('Video', 'video', _filterType, (v) => setState(() => _filterType = v)),
            _filterChip('Audio', 'audio', _filterType, (v) => setState(() => _filterType = v)),
          ],
        ),
        const SizedBox(height: 16),

        // Table
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // Table header
                    Container(
                      color: const Color(0xFFF8FAFC),
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      child: const Row(
                        children: [
                          SizedBox(width: 80, child: Text('ID', style: _headerStyle)),
                          SizedBox(width: 70, child: Text('Type', style: _headerStyle)),
                          Expanded(child: Text('File Name', style: _headerStyle)),
                          SizedBox(width: 160, child: Text('Upload Time', style: _headerStyle)),
                          SizedBox(width: 90, child: Text('Ground Truth', style: _headerStyle)),
                          SizedBox(width: 90, child: Text('Status', style: _headerStyle)),
                          SizedBox(width: 140, child: Text('Actions', style: _headerStyle)),
                        ],
                      ),
                    ),
                    const Divider(height: 1, color: Color(0xFFE2E8F0)),
                    // Table rows
                    ...filtered.map((s) => _buildRow(s)),
                    if (filtered.isEmpty)
                      const Padding(
                        padding: EdgeInsets.all(40),
                        child: Text('No samples match filters', style: TextStyle(color: Color(0xFF94A3B8))),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRow(MockSample s) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFF1F5F9))),
      ),
      child: Row(
        children: [
          SizedBox(width: 80, child: Text(s.id, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF475569), fontFamily: 'monospace'))),
          SizedBox(width: 70, child: _typeBadge(s.fileType)),
          Expanded(child: Text(s.fileName, style: const TextStyle(fontSize: 13, color: Color(0xFF1E293B)), overflow: TextOverflow.ellipsis)),
          SizedBox(width: 160, child: Text(s.uploadTime, style: const TextStyle(fontSize: 12, color: Color(0xFF94A3B8)))),
          SizedBox(width: 90, child: _truthBadge(s.groundTruth)),
          SizedBox(width: 90, child: _statusBadge(s.status)),
          SizedBox(
            width: 140,
            child: Row(
              children: [
                if (s.status == 'pending') ...[
                  _actionBtn(Icons.check_rounded, const Color(0xFF22C55E), () => _updateStatus(s.id, 'approved')),
                  const SizedBox(width: 4),
                  _actionBtn(Icons.close_rounded, const Color(0xFFEF4444), () => _updateStatus(s.id, 'rejected')),
                  const SizedBox(width: 4),
                ],
                _actionBtn(Icons.visibility_outlined, const Color(0xFF6366F1), () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Preview: ${s.fileName}'), duration: const Duration(seconds: 1)),
                  );
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionBtn(IconData icon, Color color, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.all(6),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Icon(icon, size: 16, color: color),
      ),
    );
  }

  Widget _typeBadge(String type) {
    final config = {
      'image': ('🖼️', const Color(0xFF8B5CF6)),
      'video': ('🎬', const Color(0xFF3B82F6)),
      'audio': ('🎙️', const Color(0xFFF97316)),
    };
    final (emoji, color) = config[type] ?? ('📄', Colors.grey);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text('$emoji ${type[0].toUpperCase()}${type.substring(1)}',
        style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: color),
      ),
    );
  }

  Widget _statusBadge(String status) {
    final color = status == 'approved' ? const Color(0xFF22C55E) : status == 'rejected' ? const Color(0xFFEF4444) : const Color(0xFFF59E0B);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(status[0].toUpperCase() + status.substring(1),
        style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: color),
      ),
    );
  }

  Widget _truthBadge(String truth) {
    final color = truth == 'True' ? const Color(0xFF22C55E) : truth == 'False' ? const Color(0xFFEF4444) : const Color(0xFF94A3B8);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Text(truth,
        style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: color),
      ),
    );
  }

  Widget _badge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color)),
    );
  }

  Widget _filterChip(String label, String value, String current, ValueChanged<String> onSelect) {
    final active = value == current;
    return Padding(
      padding: const EdgeInsets.only(right: 6),
      child: InkWell(
        onTap: () => onSelect(value),
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: active ? const Color(0xFF1E293B) : Colors.white.withValues(alpha: 0.6),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: active ? Colors.transparent : const Color(0xFFE2E8F0)),
          ),
          child: Text(label,
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: active ? Colors.white : const Color(0xFF64748B)),
          ),
        ),
      ),
    );
  }

  static const _headerStyle = TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Color(0xFF94A3B8), letterSpacing: 0.5);
}
