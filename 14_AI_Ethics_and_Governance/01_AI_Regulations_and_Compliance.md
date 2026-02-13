# AI Regulations and Compliance - 2025 Global Framework

## Overview

**AI governance is now mandatory.** The EU AI Act (2024), GDPR, and emerging global regulations create legal requirements for AI development and deployment.

**Key Dates:**
- **EU AI Act:** Entered force Aug 1, 2024; Full compliance by Aug 2, 2026
- **Prohibitions:** Active Feb 2, 2025
- **GPAI obligations:** Active Aug 2, 2025

---

## EU AI Act - The Global Standard

### Risk-Based Classification

**Prohibited AI (Ban effective Feb 2, 2025):**
- Social scoring by governments
- Real-time biometric identification in public (with exceptions)
- Emotion recognition in workplaces/schools
- Exploitative AI (children, vulnerable persons)
- Subliminal manipulation

**High-Risk AI (Strict requirements):**
- Critical infrastructure
- Education/vocational training
- Employment decisions
- Essential services (credit scoring, insurance)
- Law enforcement
- Migration/border control
- Justice/democratic processes
- Biometric identification

**Limited Risk (Transparency required):**
- Chatbots (must disclose AI nature)
- Deepfakes (must label)
- Emotion recognition systems

**Minimal Risk (No obligations):**
- AI-enabled video games
- Spam filters
- Most recommender systems

---

### Compliance Requirements for High-Risk AI

#### 1. Risk Management System
```python
class AIRiskManagement:
    """
    EU AI Act Article 9: Risk Management
    """

    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.risk_register = []

    def identify_risks(self):
        """Identify and document all risks"""
        risks = {
            'bias_discrimination': self.assess_bias_risk(),
            'data_privacy': self.assess_privacy_risk(),
            'safety': self.assess_safety_risk(),
            'accuracy': self.assess_accuracy_risk(),
            'security': self.assess_security_risk()
        }

        for risk_type, assessment in risks.items():
            self.risk_register.append({
                'type': risk_type,
                'severity': assessment['severity'],
                'likelihood': assessment['likelihood'],
                'mitigation': assessment['mitigation']
            })

        return self.risk_register

    def assess_bias_risk(self):
        """Article 10: Data governance - bias assessment"""
        # Check training data for bias
        bias_metrics = self.analyze_training_data_bias()

        return {
            'severity': 'HIGH' if bias_metrics['disparity'] > 0.2 else 'MEDIUM',
            'likelihood': bias_metrics['probability'],
            'mitigation': 'Implement bias detection and correction (Article 10)'
        }

    def continuous_monitoring(self):
        """Article 61: Post-market monitoring"""
        # Ongoing monitoring required
        monitoring_plan = {
            'frequency': 'monthly',
            'metrics': ['accuracy', 'bias', 'user_complaints'],
            'alert_thresholds': {
                'accuracy_drop': 0.05,
                'bias_increase': 0.1
            }
        }
        return monitoring_plan
```

#### 2. Data Governance (Article 10)

**Training Data Requirements:**

```python
class DataGovernanceCompliance:
    """
    EU AI Act Article 10: Data and Data Governance
    """

    def validate_training_data(self, dataset):
        """Ensure compliance with Article 10"""

        validation_report = {}

        # 1. Relevance
        validation_report['relevance'] = self.check_relevance(dataset)

        # 2. Representativeness
        validation_report['representativeness'] = self.check_representativeness(dataset)

        # 3. Error-free (to best extent possible)
        validation_report['data_quality'] = self.check_data_quality(dataset)

        # 4. Completeness
        validation_report['completeness'] = self.check_completeness(dataset)

        # 5. Bias examination
        validation_report['bias_analysis'] = self.examine_bias(dataset)

        # 6. Special category data handling (if applicable)
        if self.contains_sensitive_data(dataset):
            validation_report['sensitive_data_justification'] = \
                self.justify_sensitive_data_use(dataset)

        return validation_report

    def check_representativeness(self, dataset):
        """Ensure dataset represents intended population"""
        demographics = self.analyze_demographics(dataset)
        population_stats = self.get_population_statistics()

        # Compare dataset to actual population
        representativeness_score = {}
        for category in demographics:
            dataset_pct = demographics[category]
            population_pct = population_stats[category]

            # Allow +/-5% deviation
            is_representative = abs(dataset_pct - population_pct) < 0.05

            representativeness_score[category] = {
                'dataset': dataset_pct,
                'population': population_pct,
                'representative': is_representative
            }

        return representativeness_score

    def examine_bias(self, dataset):
        """Article 10: Examine datasets for possible biases"""
        bias_report = {}

        # Check for statistical parity across protected attributes
        protected_attributes = ['gender', 'race', 'age', 'disability']

        for attr in protected_attributes:
            if attr in dataset.columns:
                # Calculate representation
                distribution = dataset[attr].value_counts(normalize=True)

                # Flag if any group < 10%
                underrepresented = distribution[distribution < 0.1]

                bias_report[attr] = {
                    'distribution': distribution.to_dict(),
                    'underrepresented_groups': underrepresented.to_dict(),
                    'bias_risk': 'HIGH' if len(underrepresented) > 0 else 'LOW'
                }

        return bias_report

    def justify_sensitive_data_use(self, dataset):
        """
        Article 10: Special category data only for bias detection/correction
        """
        justification = {
            'purpose': 'bias_detection_and_correction',
            'legal_basis': 'EU AI Act Article 10',
            'necessity': 'Required to identify and mitigate discriminatory outcomes',
            'safeguards': [
                'Data minimization applied',
                'Access restricted to authorized personnel',
                'Automatic deletion after bias analysis',
                'No use for other purposes'
            ]
        }
        return justification
```

#### 3. Technical Documentation (Article 11)

**Required documentation:**

```python
class TechnicalDocumentation:
    """
    EU AI Act Article 11: Technical Documentation
    """

    def generate_documentation(self):
        """Generate comprehensive technical documentation"""

        documentation = {
            # 1. General description
            'general_description': {
                'intended_purpose': self.describe_intended_purpose(),
                'risk_level': 'HIGH-RISK',
                'reasonably_foreseeable_misuse': self.identify_misuse_scenarios()
            },

            # 2. Design and development
            'design_specs': {
                'architecture': self.document_architecture(),
                'algorithms': self.describe_algorithms(),
                'data_requirements': self.specify_data_requirements(),
                'hardware_requirements': self.list_hardware_requirements()
            },

            # 3. Training methodology
            'training': {
                'training_data': {
                    'sources': self.list_data_sources(),
                    'collection_methodology': self.describe_data_collection(),
                    'labeling': self.describe_labeling_process(),
                    'preprocessing': self.document_preprocessing()
                },
                'training_procedure': {
                    'methodology': self.describe_training_methodology(),
                    'hyperparameters': self.list_hyperparameters(),
                    'validation_strategy': self.describe_validation()
                }
            },

            # 4. Performance metrics
            'performance': {
                'accuracy_metrics': self.report_accuracy(),
                'robustness_metrics': self.report_robustness(),
                'bias_metrics': self.report_bias_metrics(),
                'limitations': self.document_limitations()
            },

            # 5. Human oversight
            'human_oversight': {
                'oversight_measures': self.describe_oversight_measures(),
                'intervention_mechanisms': self.list_intervention_mechanisms()
            },

            # 6. Cybersecurity
            'cybersecurity': {
                'security_measures': self.document_security_measures(),
                'vulnerability_assessment': self.assess_vulnerabilities()
            }
        }

        return documentation
```

#### 4. Transparency Obligations (Article 13)

```python
class TransparencyCompliance:
    """
    EU AI Act Article 13: Transparency and Information to Users
    """

    def generate_user_information(self):
        """Information to be provided to users"""

        user_info = {
            # 1. AI system identification
            'system_info': {
                'name': 'Credit Risk Assessment AI',
                'provider': 'ABC Bank',
                'is_high_risk': True,
                'conformity_certificate': 'EU-CERT-2025-12345'
            },

            # 2. Intended purpose and limitations
            'purpose_and_limitations': {
                'intended_purpose': '''
                    This AI system assesses credit risk for loan applications
                    by analyzing financial history, income, and other factors.
                ''',
                'limitations': [
                    'Not suitable for business loans > EUR1M',
                    'Requires manual review for edge cases',
                    'May have reduced accuracy for self-employed applicants'
                ],
                'residual_risks': [
                    'Potential bias against certain occupations',
                    'Limited data for recent immigrants'
                ]
            },

            # 3. Human oversight information
            'human_oversight': {
                'oversight_type': 'Human-in-the-loop',
                'how_to_request_review': 'Contact customer service to request human review',
                'contact': 'ai-review@abcbank.com'
            },

            # 4. Performance information
            'performance': {
                'accuracy': '92% (tested on 100,000 applications)',
                'bias_mitigation': 'Active bias monitoring and correction',
                'last_updated': '2025-01-15'
            }
        }

        return user_info

    def generate_deepfake_disclosure(self):
        """Article 50: Transparency for deepfakes"""
        return {
            'disclosure_required': True,
            'disclosure_text': '''
                NOTICE: This image/video/audio has been artificially generated
                or manipulated using AI technology.
            ''',
            'machine_readable_marker': True,  # Technical means to detect
            'exceptions': 'Artistic/creative works with appropriate disclosure'
        }
```

---

## GDPR Integration

### Article 5 GDPR Principles Applied to AI

```python
class GDPRAICompliance:
    """
    GDPR principles for AI systems processing personal data
    """

    def ensure_lawfulness(self):
        """Article 6 GDPR: Legal basis for processing"""
        legal_bases = {
            'consent': 'Explicit, informed, freely given (GDPR Art 6(1)(a))',
            'contract': 'Necessary for contract performance (GDPR Art 6(1)(b))',
            'legal_obligation': 'Compliance with legal obligation (GDPR Art 6(1)(c))',
            'legitimate_interest': 'Legitimate interest (GDPR Art 6(1)(f)) - requires balancing test'
        }

        # For AI, often need multiple bases
        return legal_bases

    def ensure_fairness_transparency(self):
        """Article 5(1)(a) GDPR: Fairness and transparency"""
        requirements = {
            'fairness': {
                'no_discrimination': 'Ensure AI does not discriminate',
                'no_deception': 'Transparent about AI decision-making',
                'data_subject_rights': 'Enable exercise of rights'
            },
            'transparency': {
                'information_provided': 'Clear info about AI processing (Art 13/14)',
                'right_to_explanation': 'Meaningful information about logic (Art 15)',
                'automated_decision_info': 'Specific info for automated decisions (Art 22)'
            }
        }
        return requirements

    def implement_data_minimization(self):
        """Article 5(1)(c) GDPR: Data minimization"""
        return {
            'collect_only_necessary': 'Only data adequate and relevant for AI purpose',
            'feature_selection': 'Remove unnecessary features',
            'aggregation': 'Aggregate where possible',
            'pseudonymization': 'Pseudonymize personal data (Art 32)',
            'retention': 'Delete data when no longer needed (Art 17)'
        }

    def ensure_accuracy(self):
        """Article 5(1)(d) GDPR: Accuracy"""
        return {
            'data_quality': 'Ensure training data accuracy',
            'model_accuracy': 'Monitor model performance',
            'right_to_rectification': 'Enable data subjects to correct inaccurate data (Art 16)',
            'regular_updates': 'Update model to maintain accuracy'
        }

    def implement_automated_decision_safeguards(self):
        """Article 22 GDPR: Automated decision-making"""

        # Article 22: Right not to be subject to automated decision
        safeguards = {
            'general_rule': 'Automated decisions with legal/significant effects are prohibited',
            'exceptions': [
                'Necessary for contract (Art 22(2)(a))',
                'Authorized by law (Art 22(2)(b))',
                'Based on explicit consent (Art 22(2)(c))'
            ],
            'safeguards_required': [
                'Right to obtain human intervention',
                'Right to express point of view',
                'Right to contest decision'
            ],
            'special_category_prohibition': '''
                Cannot use special category data (race, health, etc.)
                for automated decisions unless:
                - Explicit consent (Art 9(2)(a)), OR
                - Substantial public interest with safeguards (Art 9(2)(g))
            '''
        }

        return safeguards
```

---

## Global AI Regulations

### United States

```python
us_ai_regulations = {
    'federal_level': {
        'AI_Bill_of_Rights': {
            'status': 'Blueprint (non-binding)',
            'key_principles': [
                'Safe and effective systems',
                'Algorithmic discrimination protections',
                'Data privacy',
                'Notice and explanation',
                'Human alternatives and fallback'
            ]
        },
        'Executive_Order_14110': {
            'date': 'October 30, 2023',
            'key_requirements': [
                'Safety testing for powerful AI models',
                'Red-teaming requirements',
                'Reporting to government',
                'Standards development (NIST)'
            ]
        },
        'sector_specific': {
            'finance': 'Fair lending laws, ECOA, FCRA',
            'healthcare': 'HIPAA, FDA regulations for AI/ML medical devices',
            'employment': 'EEOC guidelines on AI hiring',
            'credit': 'FCRA adverse action notices'
        }
    },
    'state_level': {
        'California': {
            'CPRA': 'California Privacy Rights Act - GDPR-like',
            'CCPA': 'Foundational privacy law',
            'automated_decision_rights': 'Right to opt-out of automated decisions'
        },
        'New_York': {
            'NYC_Bias_Audit_Law': 'Mandatory bias audits for hiring AI (2023)',
            'requirements': [
                'Annual bias audit',
                'Public disclosure of audit results',
                'Notice to candidates/employees'
            ]
        },
        'Illinois': {
            'BIPA': 'Biometric Information Privacy Act - strictest in US',
            'requirements': 'Consent for biometric data collection'
        }
    }
}
```

### China

```python
china_ai_regulations = {
    'Algorithm_Recommendation_Regulations': {
        'effective_date': 'March 1, 2022',
        'requirements': [
            'Algorithm filing with CAC',
            'Transparency in recommendations',
            'User rights to opt-out',
            'Prohibition on price discrimination'
        ]
    },
    'Deep_Synthesis_Regulations': {
        'effective_date': 'January 10, 2023',
        'scope': 'Deepfakes, synthetic media',
        'requirements': [
            'Labeling of synthetic content',
            'User verification',
            'Prohibition on illegal content generation'
        ]
    },
    'Generative_AI_Regulations': {
        'effective_date': 'August 15, 2023',
        'requirements': [
            'Security assessment before deployment',
            'Content filtering',
            'User data protection',
            'Prevent discrimination'
        ]
    }
}
```

---

## Compliance Implementation Checklist

### For High-Risk AI Systems (EU AI Act)

- [ ] **Risk Classification** - Confirm if system is high-risk
- [ ] **Risk Management** - Implement Article 9 risk management system
- [ ] **Data Governance** - Article 10 compliance (representative, bias-free)
- [ ] **Technical Documentation** - Article 11 comprehensive docs
- [ ] **Record-Keeping** - Article 12 automatic logging
- [ ] **Transparency** - Article 13 user information
- [ ] **Human Oversight** - Article 14 oversight measures
- [ ] **Accuracy & Robustness** - Article 15 performance requirements
- [ ] **Cybersecurity** - Article 15 security measures
- [ ] **Quality Management** - Article 17 QMS
- [ ] **Conformity Assessment** - Third-party assessment if required
- [ ] **CE Marking** - Affix CE marking
- [ ] **Registration** - Register in EU database
- [ ] **Post-Market Monitoring** - Article 61 ongoing monitoring

### For GDPR Compliance

- [ ] **Legal Basis** - Article 6 lawful basis identified
- [ ] **Legitimate Interest Assessment** - If applicable (Art 6(1)(f))
- [ ] **Data Minimization** - Collect only necessary data
- [ ] **Purpose Limitation** - Use data only for stated purpose
- [ ] **Transparency** - Privacy notice provided (Art 13/14)
- [ ] **Data Subject Rights** - Mechanisms for Art 15-22 rights
- [ ] **Special Category Data** - Article 9 safeguards if applicable
- [ ] **Automated Decisions** - Article 22 safeguards
- [ ] **DPIA** - Data Protection Impact Assessment if high-risk
- [ ] **DPO** - Data Protection Officer appointed if required
- [ ] **Security** - Article 32 technical/organizational measures
- [ ] **Breach Procedures** - Article 33/34 notification processes

---

## Penalties and Enforcement

### EU AI Act Fines

| Violation | Maximum Fine |
|-----------|-------------|
| Prohibited AI practices | EUR35M or 7% global turnover |
| Non-compliance with obligations | EUR15M or 3% global turnover |
| Incorrect/incomplete information | EUR7.5M or 1% global turnover |

### GDPR Fines

| Violation | Maximum Fine |
|-----------|-------------|
| Articles 5-22 (principles, rights, automated decisions) | EUR20M or 4% global turnover |
| Other provisions | EUR10M or 2% global turnover |

**Notable Cases:**
- Amazon: EUR746M (2021) - Processing violations
- Meta: EUR1.2B (2023) - Data transfers
- Google: EUR90M (2020) - Cookie consent

---

## Key Takeaways

1. **Compliance is mandatory** - EU AI Act + GDPR are enforceable law
2. **Risk-based approach** - Higher risk = stricter requirements
3. **Documentation is critical** - Must prove compliance
4. **Bias mitigation required** - Not optional for high-risk AI
5. **Human oversight essential** - Cannot fully automate high-risk decisions
6. **Transparency non-negotiable** - Users must know AI is involved
7. **Global coordination** - Regulations increasingly aligned
8. **Enforcement is real** - Billion-euro fines possible

**Next:** `02_Bias_Detection_and_Mitigation.md` - Technical implementation of fairness requirements
