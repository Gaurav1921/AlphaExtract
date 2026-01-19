"""
Fix Anomaly Detection Error Handling
------------------------------------
Updates dashboard to handle error cases properly.
"""

from pathlib import Path

dashboard_file = Path('dashboard/app.py')
content = dashboard_file.read_text(encoding='utf-8')

# Fix the anomaly detection button handler
old_code = """        if st.button("🔍 Run Anomaly Detection Now"):
            with st.spinner("Analyzing..."):
                report = st.session_state.anomaly_detector.analyze_ticker(selected_ticker)
                st.session_state.anomaly_detector.save_report(report)
                
                # Save to database
                if st.session_state.db.client and report.get('anomalies'):
                    st.session_state.db.insert_anomalies(
                        ticker=selected_ticker,
                        filing_date=report['current_filing_date'],
                        anomalies=report['anomalies']
                    )
                
                st.success("Analysis complete!")
                st.rerun()"""

new_code = """        if st.button("🔍 Run Anomaly Detection Now"):
            with st.spinner("Analyzing..."):
                report = st.session_state.anomaly_detector.analyze_ticker(selected_ticker)
                
                # Check if analysis was successful
                if 'error' in report:
                    st.error(f"Analysis failed: {report['error']}")
                    st.info("💡 Tip: You need at least 2 filings to detect anomalies. Download more filings for this company.")
                else:
                    st.session_state.anomaly_detector.save_report(report)
                    
                    # Save to database
                    if st.session_state.db.client and report.get('anomalies'):
                        st.session_state.db.insert_anomalies(
                            ticker=selected_ticker,
                            filing_date=report['current_filing_date'],
                            anomalies=report['anomalies']
                        )
                    
                    st.success("Analysis complete!")
                    st.rerun()"""

content = content.replace(old_code, new_code)

dashboard_file.write_text(content, encoding='utf-8')
print("✓ Fixed anomaly detection error handling")
print("\nRestart dashboard: streamlit run dashboard/app.py")