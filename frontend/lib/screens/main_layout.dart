import 'package:flutter/material.dart';
import 'home_screen.dart'; // The current Scanner tab
import 'live_screen.dart'; // The Live Capture tab
import '../widgets/glass_bottom_nav.dart';
import '../widgets/background.dart';
import 'admin/admin_dashboard_screen.dart' as admin;

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int _currentIndex = 0;

  // The 3 main views
  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _pages = [
      HomeScreen(
        isEmbedded: true,
        onNavigateToLive: () {
          setState(() {
            _currentIndex = 1;
          });
        },
      ), // We'll modify HomeScreen to remove its own Scaffold/Background if embedded
      const LiveScreen(), // The Live Capture screen
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true, // Allow body to flow under the floating nav bar
      body: Stack(
        children: [
          // Global Background
          const AnimatedGradientBackground(
            child: SizedBox.expand(),
          ),

          // Main Content Area with Smooth Fade Transition
          Positioned.fill(
            // Add top padding so content doesn't hide behind the top nav bar
            top: 32, 
            bottom: 0,
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              switchInCurve: Curves.easeOutCubic,
              switchOutCurve: Curves.easeInCubic,
              transitionBuilder: (child, animation) {
                return FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween<Offset>(
                      begin: const Offset(0.0, 0.05),
                      end: Offset.zero,
                    ).animate(animation),
                    child: child,
                  ),
                );
              },
              child: KeyedSubtree(
                key: ValueKey<int>(_currentIndex),
                child: _pages[_currentIndex],
              ),
            ),
          ),

          // Top Left Logo
          Positioned(
            top: 24,
            left: 24,
            // Constrain the width of the logo area so it doesn't overlap with the center content
            width: 150, 
            child: SafeArea(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Logo image from assets
                  Container(
                    width: 32, // Reduced from 38
                    height: 32, // Reduced from 38
                    decoration: const BoxDecoration(
                      color: Colors.transparent,
                      shape: BoxShape.circle,
                      boxShadow: [
                         BoxShadow(color: Colors.black12, blurRadius: 8, offset: Offset(0, 4)),
                      ],
                      image: DecorationImage(
                        image: AssetImage('assets/images/Realytic.png'),
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10), // Reduced spacing slightly
                  // Wrap text in Expanded so it truncates instead of overflowing/overlapping
                  const Expanded(
                    child: Text(
                      'REALYTIC',
                      style: TextStyle(
                        fontSize: 20, // Reduced from 24
                        fontWeight: FontWeight.w900, // Maximum boldness
                        letterSpacing: -0.5, // Adjusted letter spacing for smaller size
                        color: Colors.black87,
                      ),
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Top Right Buttons (Settings & Save/History)
          Positioned(
            top: 24,
            right: 24,
            child: SafeArea(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Privacy Settings Button (Light grey circle)
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.05),
                      shape: BoxShape.circle,
                    ),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        borderRadius: BorderRadius.circular(18),
                        onTap: () => _showPrivacyDialog(context),
                        child: const Icon(
                          Icons.shield_outlined,
                          size: 20,
                          color: Colors.black54,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Settings Button (Light grey circle)
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.05),
                      shape: BoxShape.circle,
                    ),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        borderRadius: BorderRadius.circular(18),
                        onTap: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(builder: (_) => const admin.AdminDashboardScreen()),
                          );
                        },
                        child: const Icon(
                          Icons.settings_outlined,
                          size: 20,
                          color: Colors.black54,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Save/History Button (Dark pill)
                  Container(
                    height: 36,
                    decoration: BoxDecoration(
                      color: const Color(0xFF2D2D2D), // Dark grey/black
                      borderRadius: BorderRadius.circular(18),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.15),
                          blurRadius: 8,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        borderRadius: BorderRadius.circular(18),
                        onTap: () {
                          // TODO: Implement save/history action
                        },
                        child: const Padding(
                          padding: EdgeInsets.symmetric(horizontal: 16),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.history_rounded, // History icon
                                size: 18,
                                color: Colors.white,
                              ),
                              SizedBox(width: 6),
                              Text(
                                'History',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  letterSpacing: 0.2,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Floating Top Navigation Bar
          Positioned(
            left: 0,
            right: 0,
            top: 24, // Moved to top to align with logo and buttons
            child: Center(
              child: GlassBottomNav(
                currentIndex: _currentIndex,
                onTap: (index) {
                  setState(() {
                    _currentIndex = index;
                  });
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showPrivacyDialog(BuildContext context) {
    bool localConsent = true;
    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              backgroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              title: const Row(
                children: [
                  Icon(Icons.shield_outlined, color: Colors.blueAccent),
                  SizedBox(width: 10),
                  Text('Data Storage & Privacy', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
                ],
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                   const Text(
                    'To improve our models, we collect anonymized scan data. You can opt out at any time.',
                    style: TextStyle(fontSize: 13, color: Colors.black87),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.02),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.black.withValues(alpha: 0.05))
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SizedBox(
                          width: 24,
                          height: 24,
                          child: Checkbox(
                            value: localConsent,
                            onChanged: (val) => setState(() => localConsent = val ?? false),
                            activeColor: Colors.blueAccent,
                            checkColor: Colors.white,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.only(top: 2),
                            child: RichText(
                              text: const TextSpan(
                                style: TextStyle(fontSize: 13, color: Colors.black87, height: 1.4),
                                children: [
                                  TextSpan(text: 'I allow my scan data to be securely stored to retrain detection models. Read our '),
                                  TextSpan(text: 'Terms & Conditions', style: TextStyle(color: Colors.blue, fontWeight: FontWeight.bold)),
                                  TextSpan(text: ' for full details.'),
                                ]
                              )
                            )
                          )
                        )
                      ]
                    ),
                  )
                ]
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Save Preferences', style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold)),
                )
              ],
            );
          }
        );
      }
    );
  }
}

class _PlaceholderPage extends StatelessWidget {
  final String title;
  final IconData icon;

  const _PlaceholderPage({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 64, color: Colors.black26),
            const SizedBox(height: 16),
            Text(
              title,
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: Colors.black45,
                letterSpacing: -0.5,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Coming soon',
              style: TextStyle(color: Colors.black38),
            ),
          ],
        ),
      ),
    );
  }
}
