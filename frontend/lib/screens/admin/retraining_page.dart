import 'package:flutter/material.dart';
import 'mock_data.dart';

class RetrainingPage extends StatefulWidget {
  const RetrainingPage({super.key});

  @override
  State<RetrainingPage> createState() => _RetrainingPageState();
}

class _RetrainingPageState extends State<RetrainingPage> {
  String? _selectedModel;
  bool _isTraining = false;
  int _currentStepIndex = -1;

  void _startTraining() async {
    if (_selectedModel == null) return;
    
    setState(() {
      _isTraining = true;
      _currentStepIndex = 0;
    });

    // Simulate training steps
    for (int i = 0; i < mockTrainingSteps.length; i++) {
      if (!mounted) return;
      setState(() => _currentStepIndex = i);
      
      // Simulate variable step duration
      int ms = i < 2 ? 800 : (i < 7 ? 1500 : 1000); 
      await Future.delayed(Duration(milliseconds: ms));
    }

    if (mounted) {
      setState(() {
        _isTraining = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Retraining completed. Candidate model saved.'),
          backgroundColor: Color(0xFF22C55E),
        ),
      );
    }
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
              Icon(Icons.model_training_outlined, color: Color(0xFF6366F1), size: 22),
              SizedBox(width: 10),
              Text('Model Retraining', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B))),
            ],
          ),
          const SizedBox(height: 8),
          const Text('Select a model architecture to retrain using the latest approved dataset.',
            style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
          ),
          const SizedBox(height: 28),

          // Model Selection Grid
          const Text('Target Model', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
          const SizedBox(height: 12),
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 16,
              mainAxisSpacing: 16,
              childAspectRatio: 2.5,
            ),
            itemCount: mockModelCards.length,
            itemBuilder: (context, index) {
              final model = mockModelCards[index];
              final isSelected = _selectedModel == model.name;
              
              return InkWell(
                onTap: _isTraining ? null : () => setState(() => _selectedModel = model.name),
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: isSelected ? const Color(0xFF6366F1).withValues(alpha: 0.05) : Colors.white.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isSelected ? const Color(0xFF6366F1) : Colors.black.withValues(alpha: 0.06),
                      width: isSelected ? 2 : 1,
                    ),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(model.icon, style: const TextStyle(fontSize: 24)),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(model.name, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
                            const SizedBox(height: 4),
                            Text(model.description, style: const TextStyle(fontSize: 12, color: Color(0xFF64748B)), maxLines: 1, overflow: TextOverflow.ellipsis),
                            const Spacer(),
                            Row(
                              children: [
                                _miniBadge('v${model.currentVersion}', Colors.grey.shade700),
                                const SizedBox(width: 8),
                                _miniBadge('${model.trainingSamples} samples', const Color(0xFF6366F1)),
                              ],
                            )
                          ],
                        ),
                      ),
                      if (isSelected)
                        const Icon(Icons.check_circle_rounded, color: Color(0xFF6366F1), size: 20),
                    ],
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: 32),

          // Start Button & Progress Area
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      _selectedModel != null ? 'Retraining: $_selectedModel' : 'Select a model to begin',
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1E293B)),
                    ),
                    ElevatedButton.icon(
                      onPressed: (_selectedModel == null || _isTraining) ? null : _startTraining,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF6366F1),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      icon: const Icon(Icons.play_arrow_rounded, size: 20),
                      label: Text(_isTraining ? 'Training in Progress...' : 'Start Retraining'),
                    ),
                  ],
                ),
                
                if (_isTraining || _currentStepIndex >= 0) ...[
                  const SizedBox(height: 24),
                  const Divider(color: Color(0xFFE2E8F0)),
                  const SizedBox(height: 16),
                  
                  // Progress Bar
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _currentStepIndex >= 0 ? (_currentStepIndex + 1) / mockTrainingSteps.length : 0,
                      backgroundColor: const Color(0xFFF1F5F9),
                      valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF6366F1)),
                      minHeight: 8,
                    ),
                  ),
                  const SizedBox(height: 24),
                  
                  // Logs Terminal
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0F172A),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFF334155)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: List.generate(mockTrainingSteps.length, (index) {
                        final step = mockTrainingSteps[index];
                        
                        // Determine visual status based on simulation
                        String statusStr = 'pending';
                        Color statusColor = const Color(0xFF64748B);
                        IconData statusIcon = Icons.hourglass_empty;
                        
                        if (_currentStepIndex > index) {
                          statusStr = 'done';
                          statusColor = const Color(0xFF22C55E);
                          statusIcon = Icons.check_circle_outline;
                        } else if (_currentStepIndex == index && _isTraining) {
                          statusStr = 'running';
                          statusColor = const Color(0xFF3B82F6);
                          statusIcon = Icons.sync;
                        }

                        // Opacity effect for pending
                        final isPending = statusStr == 'pending';
                        
                        return Opacity(
                          opacity: isPending ? 0.4 : 1.0,
                          child: Padding(
                            padding: const EdgeInsets.only(bottom: 8.0),
                            child: Row(
                              children: [
                                _isTraining && _currentStepIndex == index 
                                  ? const SizedBox(
                                      width: 14, 
                                      height: 14, 
                                      child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF3B82F6))
                                    )
                                  : Icon(statusIcon, size: 14, color: statusColor),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: Text(
                                    step['step'] as String,
                                    style: const TextStyle(color: Colors.white, fontSize: 13, fontFamily: 'monospace'),
                                  ),
                                ),
                                Text(
                                  statusStr == 'done' && _currentStepIndex > index ? step['duration'] as String : (statusStr == 'running' ? '...' : ''),
                                  style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12, fontFamily: 'monospace'),
                                ),
                              ],
                            ),
                          ),
                        );
                      }),
                    ),
                  ),
                ]
              ],
            ),
          )
        ],
      ),
    );
  }

  Widget _miniBadge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Text(text, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: color)),
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
