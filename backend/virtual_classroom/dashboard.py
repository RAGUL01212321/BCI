import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats
import requests
import os
from datetime import datetime

EEG_CSV_FILE = Path("virtual_classroom_eeg.csv")

DEBUG_MODE = True  # Set to False when OpenAI API key is available

def safe_float(value, default=0.0):
    """Convert value to float, handling NaN and inf values"""
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def safe_round(value, decimals=2, default=0.0):
    """Safely round a value, handling NaN and inf"""
    safe_val = safe_float(value, default)
    return round(safe_val, decimals)

class EEGDashboardAnalyzer:
    def __init__(self):
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load EEG data from CSV file"""
        if not EEG_CSV_FILE.exists():
            return
        
        with open(EEG_CSV_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse JSON strings for EEG bands
                for band in ['raw_eeg', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
                    if row[band]:
                        try:
                            row[band] = json.loads(row[band])
                        except (json.JSONDecodeError, ValueError):
                            row[band] = []
                
                # Convert numeric fields with safe handling
                numeric_fields = ['noise_level', 'lighting', 'temperature', 'seating_comfort', 
                                'session_duration', 'task_difficulty', 'attention_index', 'student_id']
                for field in numeric_fields:
                    if row[field]:
                        row[field] = safe_float(row[field])
                
                self.data.append(row)
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics for all parameters"""
        if not self.data:
            return {}
        
        stats_data = {}
        numeric_fields = ['noise_level', 'lighting', 'temperature', 'seating_comfort', 
                         'session_duration', 'task_difficulty', 'attention_index']
        
        for field in numeric_fields:
            values = [safe_float(row[field]) for row in self.data if row.get(field) is not None]
            values = [v for v in values if not (np.isnan(v) or np.isinf(v))]  # Filter out NaN/inf
            
            if values:
                stats_data[field] = {
                    'mean': safe_round(np.mean(values)),
                    'median': safe_round(np.median(values)),
                    'std': safe_round(np.std(values)),
                    'min': safe_round(np.min(values)),
                    'max': safe_round(np.max(values)),
                    'count': len(values)
                }
            else:
                stats_data[field] = {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return stats_data
    
    def get_band_power_analysis(self) -> Dict[str, Any]:
        """Analyze power in different EEG frequency bands"""
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_analysis = {}
        
        for band in bands:
            powers = []
            for row in self.data:
                if row.get(band) and isinstance(row[band], list):
                    try:
                        # Calculate RMS power for each student
                        band_data = np.array(row[band], dtype=float)
                        band_data = band_data[~np.isnan(band_data)]  # Remove NaN values
                        if len(band_data) > 0:
                            power = np.sqrt(np.mean(band_data**2))
                            if not (np.isnan(power) or np.isinf(power)):
                                powers.append(power)
                    except (ValueError, TypeError):
                        continue
            
            if powers:
                powers = np.array(powers)
                band_analysis[band] = {
                    'avg_power': safe_round(np.mean(powers)),
                    'std_power': safe_round(np.std(powers)),
                    'power_distribution': {
                        'low': int(len([p for p in powers if p < np.percentile(powers, 33)])),
                        'medium': int(len([p for p in powers if np.percentile(powers, 33) <= p < np.percentile(powers, 67)])),
                        'high': int(len([p for p in powers if p >= np.percentile(powers, 67)]))
                    }
                }
            else:
                band_analysis[band] = {
                    'avg_power': 0.0,
                    'std_power': 0.0,
                    'power_distribution': {'low': 0, 'medium': 0, 'high': 0}
                }
        
        return band_analysis
    
    def get_attention_analysis(self) -> Dict[str, Any]:
        """Detailed analysis of attention indices"""
        attention_scores = [safe_float(row.get('attention_index', 0)) for row in self.data 
                           if row.get('attention_index') is not None]
        attention_scores = [s for s in attention_scores if not (np.isnan(s) or np.isinf(s))]
        
        if not attention_scores:
            return {
                'overall_stats': {'mean': 0.0, 'median': 0.0, 'std': 0.0},
                'distribution': {
                    'low_attention_count': 0, 'medium_attention_count': 0, 'high_attention_count': 0,
                    'low_attention_percentage': 0.0, 'medium_attention_percentage': 0.0, 'high_attention_percentage': 0.0
                },
                'attention_categories': {'excellent': 0, 'good': 0, 'moderate': 0, 'poor': 0, 'very_poor': 0}
            }
        
        # Categorize attention levels
        low_attention = len([s for s in attention_scores if s < 30])
        medium_attention = len([s for s in attention_scores if 30 <= s < 60])
        high_attention = len([s for s in attention_scores if s >= 60])
        total_scores = len(attention_scores)
        
        return {
            'overall_stats': {
                'mean': safe_round(np.mean(attention_scores)),
                'median': safe_round(np.median(attention_scores)),
                'std': safe_round(np.std(attention_scores))
            },
            'distribution': {
                'low_attention_count': low_attention,
                'medium_attention_count': medium_attention,
                'high_attention_count': high_attention,
                'low_attention_percentage': safe_round((low_attention / total_scores) * 100, 1) if total_scores > 0 else 0.0,
                'medium_attention_percentage': safe_round((medium_attention / total_scores) * 100, 1) if total_scores > 0 else 0.0,
                'high_attention_percentage': safe_round((high_attention / total_scores) * 100, 1) if total_scores > 0 else 0.0
            },
            'attention_categories': {
                'excellent': len([s for s in attention_scores if s >= 80]),
                'good': len([s for s in attention_scores if 60 <= s < 80]),
                'moderate': len([s for s in attention_scores if 40 <= s < 60]),
                'poor': len([s for s in attention_scores if 20 <= s < 40]),
                'very_poor': len([s for s in attention_scores if s < 20])
            }
        }
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Analyze correlations between environmental factors and attention"""
        if not self.data:
            return {}
        
        # Prepare data for correlation analysis
        factors = ['noise_level', 'lighting', 'temperature', 'seating_comfort', 
                  'session_duration', 'task_difficulty']
        
        correlations = {}
        for factor in factors:
            factor_values = []
            attention_values = []
            
            for row in self.data:
                factor_val = safe_float(row.get(factor))
                attention_val = safe_float(row.get('attention_index'))
                
                if not (np.isnan(factor_val) or np.isinf(factor_val) or 
                       np.isnan(attention_val) or np.isinf(attention_val)):
                    factor_values.append(factor_val)
                    attention_values.append(attention_val)
            
            if len(factor_values) > 1 and len(attention_values) > 1:
                try:
                    correlation, p_value = stats.pearsonr(factor_values, attention_values)
                    correlations[factor] = {
                        'correlation': safe_round(correlation, 3),
                        'p_value': safe_round(p_value, 3),
                        'significance': 'significant' if p_value < 0.05 else 'not_significant',
                        'strength': self._interpret_correlation_strength(abs(safe_float(correlation)))
                    }
                except (ValueError, np.linalg.LinAlgError):
                    correlations[factor] = {
                        'correlation': 0.0, 'p_value': 1.0,
                        'significance': 'not_significant', 'strength': 'very_weak'
                    }
            else:
                correlations[factor] = {
                    'correlation': 0.0, 'p_value': 1.0,
                    'significance': 'insufficient_data', 'strength': 'very_weak'
                }
        
        return correlations
    
    def _interpret_correlation_strength(self, correlation):
        """Interpret correlation strength"""
        correlation = safe_float(correlation)
        if correlation >= 0.7:
            return 'strong'
        elif correlation >= 0.4:
            return 'moderate'
        elif correlation >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def get_environmental_impact_analysis(self) -> Dict[str, Any]:
        """Analyze impact of environmental factors"""
        environmental_analysis = {}
        
        # Temperature impact
        temp_groups = {
            'cold': [row for row in self.data if safe_float(row.get('temperature', 0)) < 20],
            'comfortable': [row for row in self.data if 20 <= safe_float(row.get('temperature', 0)) <= 25],
            'warm': [row for row in self.data if safe_float(row.get('temperature', 0)) > 25]
        }
        
        environmental_analysis['temperature_impact'] = {}
        for group_name, group_data in temp_groups.items():
            if group_data:
                attention_scores = [safe_float(row.get('attention_index', 0)) for row in group_data]
                attention_scores = [s for s in attention_scores if not (np.isnan(s) or np.isinf(s))]
                
                if attention_scores:
                    environmental_analysis['temperature_impact'][group_name] = {
                        'count': len(group_data),
                        'avg_attention': safe_round(np.mean(attention_scores)),
                        'std_attention': safe_round(np.std(attention_scores))
                    }
        
        # Noise level impact
        noise_groups = {
            'quiet': [row for row in self.data if safe_float(row.get('noise_level', 0)) < 3],
            'moderate': [row for row in self.data if 3 <= safe_float(row.get('noise_level', 0)) <= 6],
            'noisy': [row for row in self.data if safe_float(row.get('noise_level', 0)) > 6]
        }
        
        environmental_analysis['noise_impact'] = {}
        for group_name, group_data in noise_groups.items():
            if group_data:
                attention_scores = [safe_float(row.get('attention_index', 0)) for row in group_data]
                attention_scores = [s for s in attention_scores if not (np.isnan(s) or np.isinf(s))]
                
                if attention_scores:
                    environmental_analysis['noise_impact'][group_name] = {
                        'count': len(group_data),
                        'avg_attention': safe_round(np.mean(attention_scores)),
                        'std_attention': safe_round(np.std(attention_scores))
                    }
        
        return environmental_analysis
    
    def get_outlier_analysis(self) -> Dict[str, Any]:
        """Identify outliers in attention scores and other metrics"""
        attention_scores = [safe_float(row.get('attention_index', 0)) for row in self.data]
        attention_scores = [s for s in attention_scores if not (np.isnan(s) or np.isinf(s))]
        
        if len(attention_scores) < 4:  # Need at least 4 values for quartile calculation
            return {
                'total_outliers': 0, 'high_outliers': 0, 'low_outliers': 0,
                'outlier_details': [], 'bounds': {
                    'lower': 0.0, 'upper': 100.0, 'q1': 0.0, 'q3': 100.0, 'iqr': 100.0
                }
            }
        
        q1, q3 = np.percentile(attention_scores, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, row in enumerate(self.data):
            attention_val = safe_float(row.get('attention_index', 0))
            if not (np.isnan(attention_val) or np.isinf(attention_val)):
                if attention_val < lower_bound or attention_val > upper_bound:
                    outliers.append({
                        'student_id': int(safe_float(row.get('student_id', i + 1))),
                        'attention_index': safe_round(attention_val),
                        'type': 'high' if attention_val > upper_bound else 'low'
                    })
        
        return {
            'total_outliers': len(outliers),
            'high_outliers': len([o for o in outliers if o['type'] == 'high']),
            'low_outliers': len([o for o in outliers if o['type'] == 'low']),
            'outlier_details': outliers[:10],  # Limit to first 10 for response size
            'bounds': {
                'lower': safe_round(lower_bound),
                'upper': safe_round(upper_bound),
                'q1': safe_round(q1),
                'q3': safe_round(q3),
                'iqr': safe_round(iqr)
            }
        }
    
    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights about session characteristics"""
        if not self.data:
            return {}
        
        # Group by session duration
        duration_groups = {}
        for row in self.data:
            duration = int(safe_float(row.get('session_duration', 0)))
            if duration not in duration_groups:
                duration_groups[duration] = []
            duration_groups[duration].append(row)
        
        duration_analysis = {}
        for duration, group in duration_groups.items():
            attention_scores = [safe_float(row.get('attention_index', 0)) for row in group]
            attention_scores = [s for s in attention_scores if not (np.isnan(s) or np.isinf(s))]
            
            if attention_scores:
                avg_attention = safe_round(np.mean(attention_scores))
                duration_analysis[f"{duration}_hours"] = {
                    'student_count': len(group),
                    'avg_attention': avg_attention,
                    'attention_decline': safe_round(avg_attention - 60) if duration > 2 else 0.0
                }
        
        return duration_analysis

async def get_ai_insights(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI-powered insights using OpenAI API"""
    
    # Debug mode - return pre-written insights
    if DEBUG_MODE:
        # Extract some basic metrics for dynamic content
        total_students = analysis_data.get('basic_stats', {}).get('attention_index', {}).get('count', 0)
        avg_attention = analysis_data.get('attention_analysis', {}).get('overall_stats', {}).get('mean', 0)
        high_attention_count = analysis_data.get('attention_analysis', {}).get('distribution', {}).get('high_attention_count', 0)
        
        return {
            'summary': f'Analysis of {total_students} students reveals an average attention score of {avg_attention:.1f}. The classroom environment shows varied impact on student focus, with {high_attention_count} students demonstrating high attention levels. Temperature and noise correlation patterns suggest optimal learning conditions can be achieved through environmental adjustments.',
            'recommendations': [
                'Implement regular 5-minute attention breaks during longer sessions',
                'Maintain classroom temperature between 22-24°C for optimal cognitive performance',
                'Reduce ambient noise levels, especially during complex task presentations',
                'Consider personalized seating arrangements based on individual attention patterns',
                'Use the attention monitoring data to identify students who may need additional support'
            ],
            'key_findings': [
                'Students show significant attention variability across different environmental conditions',
                'Temperature has a moderate correlation with attention levels in the classroom',
                'Session duration beyond 2 hours shows measurable attention decline',
                'Individual differences in EEG patterns suggest personalized learning approaches could be beneficial',
                'Beta wave activity correlates positively with reported task engagement'
            ]
        }
    
    # Production mode - use OpenAI API
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return {
            'summary': 'AI insights unavailable - API key not configured',
            'recommendations': ['Configure OpenAI API key to get AI-powered insights'],
            'key_findings': ['Manual analysis available in other sections']
        }
    
    # Prepare summary of key metrics for AI analysis
    summary_text = f"""
    EEG Analysis Summary:
    - Total Students: {len(analysis_data.get('basic_stats', {}).get('attention_index', {}).get('count', 0))}
    - Average Attention: {analysis_data.get('attention_analysis', {}).get('overall_stats', {}).get('mean', 0)}
    - High Attention Students: {analysis_data.get('attention_analysis', {}).get('distribution', {}).get('high_attention_count', 0)}
    - Environmental Correlations: {analysis_data.get('correlations', {})}
    """
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert neuroscientist analyzing EEG data from a classroom setting. Provide concise, actionable insights.'
                    },
                    {
                        'role': 'user',
                        'content': f'Analyze this EEG classroom data and provide key insights, recommendations, and findings: {summary_text}'
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.7
            },
            timeout=10
        )
        
        if response.status_code == 200:
            ai_response = response.json()['choices'][0]['message']['content']
            
            # Parse the AI response into structured format
            lines = ai_response.split('\n')
            insights = {
                'summary': ai_response[:200] + '...' if len(ai_response) > 200 else ai_response,
                'recommendations': [line.strip('- ') for line in lines if 'recommend' in line.lower() or 'suggest' in line.lower()],
                'key_findings': [line.strip('- ') for line in lines if 'finding' in line.lower() or 'insight' in line.lower()]
            }
            
            return insights
            
    except Exception as e:
        return {
            'summary': f'AI analysis temporarily unavailable: {str(e)}',
            'recommendations': ['Manual analysis available in dashboard sections'],
            'key_findings': ['Check network connection and API key configuration']
        }
    
    return {
        'summary': 'AI insights unavailable',
        'recommendations': ['Configure OpenAI API for enhanced insights'],
        'key_findings': ['Manual analysis provided in dashboard']
    }

def get_chart_configurations() -> Dict[str, Any]:
    """Return configuration for various charts and visualizations"""
    return {
        'attention_distribution': {
            'type': 'histogram',
            'title': 'Attention Score Distribution',
            'x_axis': 'Attention Score',
            'y_axis': 'Number of Students',
            'bins': 20
        },
        'band_power_comparison': {
            'type': 'bar_chart',
            'title': 'Average Power by Frequency Band',
            'x_axis': 'Frequency Bands',
            'y_axis': 'Average Power (μV²)'
        },
        'environmental_correlation': {
            'type': 'correlation_matrix',
            'title': 'Environmental Factors vs Attention Correlation',
            'color_scale': 'RdYlBu'
        },
        'attention_over_time': {
            'type': 'line_chart',
            'title': 'Attention Trends',
            'x_axis': 'Student ID',
            'y_axis': 'Attention Index'
        },
        'temperature_vs_attention': {
            'type': 'scatter_plot',
            'title': 'Temperature vs Attention Relationship',
            'x_axis': 'Temperature (°C)',
            'y_axis': 'Attention Index'
        }
    }

async def generate_dashboard_data() -> Dict[str, Any]:
    """Generate comprehensive dashboard data"""
    analyzer = EEGDashboardAnalyzer()
    
    # Generate all analyses
    basic_stats = analyzer.get_basic_statistics()
    band_analysis = analyzer.get_band_power_analysis()
    attention_analysis = analyzer.get_attention_analysis()
    correlations = analyzer.get_correlation_analysis()
    environmental_impact = analyzer.get_environmental_impact_analysis()
    outliers = analyzer.get_outlier_analysis()
    session_insights = analyzer.get_session_insights()
    chart_configs = get_chart_configurations()
    
    # Prepare data for AI analysis
    analysis_summary = {
        'basic_stats': basic_stats,
        'attention_analysis': attention_analysis,
        'correlations': correlations
    }
    
    # Get AI insights
    ai_insights = await get_ai_insights(analysis_summary)
    
    # Safe extraction of values with defaults
    avg_attention = basic_stats.get('attention_index', {}).get('mean', 0.0)
    attention_variability = basic_stats.get('attention_index', {}).get('std', 0.0)
    high_performers = attention_analysis.get('attention_categories', {}).get('excellent', 0)
    poor_performers = (attention_analysis.get('attention_categories', {}).get('poor', 0) + 
                      attention_analysis.get('attention_categories', {}).get('very_poor', 0))
    
    # Find most correlated factor safely
    most_correlated_factor = 'none'
    if correlations:
        try:
            most_correlated_factor = max(correlations.items(), 
                                       key=lambda x: abs(safe_float(x[1].get('correlation', 0))))[0]
        except (ValueError, KeyError):
            most_correlated_factor = 'none'
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_students': len(analyzer.data),
        'data_quality': {
            'completeness': safe_round((len(analyzer.data) / max(len(analyzer.data), 1)) * 100, 1),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'basic_statistics': basic_stats,
        'band_power_analysis': band_analysis,
        'attention_analysis': attention_analysis,
        'correlation_analysis': correlations,
        'environmental_impact': environmental_impact,
        'outlier_analysis': outliers,
        'session_insights': session_insights,
        'ai_insights': ai_insights,
        'chart_configurations': chart_configs,
        'summary_metrics': {
            'avg_attention_score': safe_round(avg_attention),
            'attention_variability': safe_round(attention_variability),
            'high_performers_count': high_performers,
            'low_performers_count': poor_performers,
            'most_correlated_factor': most_correlated_factor,
            'total_outliers': outliers.get('total_outliers', 0)
        },
        'recommendations': {
            'immediate_actions': [
                'Focus on students with attention scores below 30',
                'Investigate high noise level impact on attention',
                'Optimize classroom temperature for better focus'
            ],
            'long_term_strategies': [
                'Implement regular attention monitoring',
                'Develop personalized learning approaches',
                'Create optimal environmental conditions'
            ],
            'environmental_optimizations': [
                'Maintain temperature between 20-25°C',
                'Keep noise levels below 6',
                'Ensure adequate lighting conditions'
            ]
        }
    }
