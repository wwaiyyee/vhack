import 'package:flutter/material.dart';
import '../../widgets/background.dart';
import 'retained_samples_page.dart';
import 'dataset_builder_page.dart';
import 'pipeline_flow_screen.dart';

class AdminDashboardScreen extends StatefulWidget {
  const AdminDashboardScreen({super.key});

  @override
  State<AdminDashboardScreen> createState() => _AdminDashboardScreenState();
}

class _AdminDashboardScreenState extends State<AdminDashboardScreen> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    RetainedSamplesPage(),
    DatasetBuilderPage(),
    PipelineFlowScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Stack(
        children: [
          // Ambient Background
          const AnimatedGradientBackground(child: SizedBox.expand()),
          
          SafeArea(
            child: Row(
              children: [
                // Sidebar Navigation
                Container(
                  width: 260,
                  margin: const EdgeInsets.all(16),
                  padding: const EdgeInsets.symmetric(vertical: 24),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.7),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: Colors.white.withValues(alpha: 0.5), width: 1.5),
                    boxShadow: [
                      BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 20, offset: const Offset(0, 4)),
                    ],
                  ),
                  child: Column(
                    children: [
                      // Logo/Title
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 24),
                        child: Row(
                          children: [
                            Container(
                              width: 32, height: 32,
                              decoration: const BoxDecoration(
                                shape: BoxShape.circle,
                                image: DecorationImage(image: AssetImage('assets/images/Realytic.png'), fit: BoxFit.cover),
                              ),
                            ),
                            const SizedBox(width: 12),
                            const Text('REALYTIC', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900, letterSpacing: -0.5, color: Color(0xFF1E293B))),
                          ],
                        ),
                      ),
                      const SizedBox(height: 32),
                      
                      // Section Label
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 24),
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: Text('RETRAINING PIPELINE', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, color: Color(0xFF94A3B8), letterSpacing: 0.5)),
                        ),
                      ),
                      const SizedBox(height: 12),
                      
                      // Nav Items
                      _navItem(0, Icons.storage_rounded, 'Retained Samples'),
                      _navItem(1, Icons.dataset_outlined, 'Dataset Builder'),
                      _navItem(2, Icons.hub_outlined, 'Retraining Pipeline'),
                      
                      const Spacer(),
                      
                      // Exit text/button placeholder
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 24),
                        child: InkWell(
                          onTap: () => Navigator.of(context).pop(),
                          borderRadius: BorderRadius.circular(8),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(vertical: 8.0),
                            child: Row(
                              children: [
                                const Icon(Icons.arrow_back_rounded, size: 18, color: Color(0xFF64748B)),
                                const SizedBox(width: 12),
                                const Text('Back to App', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF64748B))),
                              ],
                            ),
                          ),
                        ),
                      )
                    ],
                  ),
                ),
                
                // Main Content Area
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(top: 16, right: 16, bottom: 16),
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      switchInCurve: Curves.easeOutCubic,
                      switchOutCurve: Curves.easeInCubic,
                      transitionBuilder: (child, animation) {
                        return FadeTransition(
                          opacity: animation,
                          child: SlideTransition(
                            position: Tween<Offset>(begin: const Offset(0.02, 0), end: Offset.zero).animate(animation),
                            child: child,
                          ),
                        );
                      },
                      child: KeyedSubtree(
                        key: ValueKey<int>(_selectedIndex),
                        child: _pages[_selectedIndex],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }

  Widget _navItem(int index, IconData icon, String title) {
    bool isSelected = _selectedIndex == index;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: InkWell(
        onTap: () => setState(() => _selectedIndex = index),
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF6366F1).withValues(alpha: 0.1) : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Icon(icon, size: 20, color: isSelected ? const Color(0xFF6366F1) : const Color(0xFF64748B)),
              const SizedBox(width: 12),
              Text(
                title,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                  color: isSelected ? const Color(0xFF6366F1) : const Color(0xFF475569),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
