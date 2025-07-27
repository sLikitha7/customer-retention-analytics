
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Silent Churn Detection Platform",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high { border-left-color: #d62728 !important; }
    .risk-medium { border-left-color: #ff7f0e !important; }
    .risk-low { border-left-color: #2ca02c !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed customer data"""
    try:
        # Try to load the final scored data first
        if os.path.exists('outputs/final_customer_data.csv'):
            df = pd.read_csv('outputs/final_customer_data.csv')
            print("✅ Loaded final scored customer data")
            return df
        # Fallback to initial processed data
        elif os.path.exists('data/processed_customers.csv'):
            df = pd.read_csv('data/processed_customers.csv')
            print("✅ Loaded initial processed customer data")
            
            # Ensure customer_status column exists
            if 'customer_status' not in df.columns:
                def initial_customer_status(row):
                    if row['frustration_score'] > 70:
                        return 'At Risk'
                    elif row['frustration_score'] > 40:
                        return 'Frustrated' 
                    else:
                        return 'Healthy'
                
                df['customer_status'] = df.apply(initial_customer_status, axis=1)
                print("⚠️ Added initial customer_status column based on frustration_score")
            
            return df
        else:
            st.error("❌ No data files found. Please run the ETL pipeline first.")
            st.info("💡 Run: python main.py")
            return None
            
    except FileNotFoundError:
        st.error("❌ Data file not found. Please run the ETL pipeline first.")
        st.info("💡 Run: python main.py")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_data
def load_recommendations():
    """Load customer recommendations"""
    try:
        if os.path.exists('outputs/recommendations.csv'):
            rec = pd.read_csv('outputs/recommendations.csv')
            return rec
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load recommendations: {str(e)}")
        return None

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚨 Silent Churn Detection & Customer Frustration Analytics")
    st.markdown("*Real-time insights into customer health and churn risk*")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Verify required columns exist
    required_columns = ['customer_status', 'frustration_score', 'mrr', 'segment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("💡 Please run the complete pipeline: python main.py")
        st.stop()
    
    recommendations = load_recommendations()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Segment filter
    segments = ['All'] + list(df['segment'].unique())
    selected_segment = st.sidebar.selectbox("Customer Segment", segments)
    
    # Status filter
    statuses = ['All'] + list(df['customer_status'].unique())
    selected_status = st.sidebar.selectbox("Customer Status", statuses)
    
    # Risk level filter
    risk_levels = ['All', 'High Risk', 'Medium Risk', 'Low Risk']
    if 'churn_risk_label' in df.columns:
        selected_risk = st.sidebar.selectbox("Churn Risk Level", risk_levels)
    else:
        selected_risk = 'All'
    
    # MRR range filter
    mrr_range = st.sidebar.slider(
        "MRR Range ($)",
        min_value=int(df['mrr'].min()),
        max_value=int(df['mrr'].max()),
        value=(int(df['mrr'].min()), int(df['mrr'].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['segment'] == selected_segment]
    
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['customer_status'] == selected_status]
    
    if selected_risk != 'All' and 'churn_risk_label' in df.columns:
        filtered_df = filtered_df[filtered_df['churn_risk_label'] == selected_risk]
    
    filtered_df = filtered_df[
        (filtered_df['mrr'] >= mrr_range[0]) & 
        (filtered_df['mrr'] <= mrr_range[1])
    ]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # KPI Cards
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Customers",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        at_risk_count = len(filtered_df[filtered_df['customer_status'] == 'At Risk'])
        at_risk_pct = (at_risk_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown('<div class="metric-card risk-high">', unsafe_allow_html=True)
        st.metric(
            label="At Risk Customers",
            value=f"{at_risk_count:,}",
            delta=f"{at_risk_pct:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_frustration = filtered_df['frustration_score'].mean()
        st.markdown('<div class="metric-card risk-medium">', unsafe_allow_html=True)
        st.metric(
            label="Avg Frustration Score",
            value=f"{avg_frustration:.1f}",
            delta=f"{avg_frustration - df['frustration_score'].mean():.1f}" if len(filtered_df) != len(df) else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        total_mrr = filtered_df['mrr'].sum()
        st.markdown('<div class="metric-card risk-low">', unsafe_allow_html=True)
        st.metric(
            label="Total MRR",
            value=f"${total_mrr:,.0f}",
            delta=f"${total_mrr - df['mrr'].sum():,.0f}" if len(filtered_df) != len(df) else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    st.header("📊 Analytics Dashboard")
    
    # Row 1: Frustration distribution and Status by Segment
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frustration Score Distribution")
        
        fig = px.histogram(
            filtered_df, 
            x='frustration_score',
            nbins=30,
            title="Distribution of Customer Frustration Scores",
            color_discrete_sequence=['skyblue']
        )
        fig.add_vline(
            x=filtered_df['frustration_score'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {filtered_df['frustration_score'].mean():.1f}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Status by Segment")
        
        # Cross-tabulation
        ct = pd.crosstab(filtered_df['segment'], filtered_df['customer_status'])
        
        fig = px.bar(
            ct,
            title="Customer Status Distribution by Segment",
            color_discrete_map={
                'Healthy': 'green',
                'Frustrated': 'orange', 
                'At Risk': 'red'
            }
        )
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: MRR vs Frustration scatter and Support metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MRR vs Frustration Analysis")
        
        fig = px.scatter(
            filtered_df,
            x='frustration_score',
            y='mrr',
            color='customer_status',
            size='tenure_months',
            hover_data=['customer_id', 'segment'],
            title="Customer Value vs. Frustration Score",
            color_discrete_map={
                'Healthy': 'green',
                'Frustrated': 'orange', 
                'At Risk': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Support Ticket Analysis")
        
        fig = px.box(
            filtered_df,
            x='segment',
            y='support_tickets_3m',
            title="Support Tickets by Segment (Last 3 Months)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Detail Section
    st.header("👥 Customer Details")
    
    # Search and sort options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search Customer ID")
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ['frustration_score', 'mrr', 'tenure_months', 'support_tickets_3m']
        )
    
    with col3:
        sort_order = st.selectbox("Order", ['Descending', 'Ascending'])
    
    # Filter and sort data for display
    display_df = filtered_df.copy()
    
    if search_term:
        display_df = display_df[
            display_df['customer_id'].str.contains(search_term, case=False, na=False)
        ]
    
    ascending = sort_order == 'Ascending'
    display_df = display_df.sort_values(sort_by, ascending=ascending)
    
    # Display customer table
    st.dataframe(
        display_df[[
            'customer_id', 'segment', 'customer_status', 'frustration_score',
            'mrr', 'tenure_months', 'support_tickets_3m', 'days_since_last_login'
        ]].head(100),
        use_container_width=True
    )
    
    # Recommendations section
    if recommendations is not None and not recommendations.empty:
        st.header("💡 Action Recommendations")
        
        # Filter recommendations based on current filters
        if selected_segment != 'All':
            rec_customers = display_df['customer_id'].tolist()
            filtered_rec = recommendations[
                recommendations['customer_id'].isin(rec_customers)
            ]
        else:
            filtered_rec = recommendations
        
        if not filtered_rec.empty:
            st.dataframe(
                filtered_rec[[
                    'customer_id', 'risk_level', 'frustration_score',
                    'primary_issues', 'recommended_actions', 'priority_score'
                ]].head(20),
                use_container_width=True
            )
            
        else:
            st.info("No specific recommendations for current filter selection.")
    else:
        st.header("💡 Action Recommendations")
        st.info("💡 No recommendations available. Run the complete pipeline (python main.py) to generate ML-powered recommendations.")
    
    # Export section
    st.header("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Customer Data", key="Download_Customer_Data"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"customer_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if recommendations is not None and not recommendations.empty:
            if st.button("💡 Download Recommendations", key="download_recommendations_export"):
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="Download Recommendations CSV",
                    data=csv,
                    file_name=f"recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("💡 Recommendations will be available after running the complete pipeline")
if __name__ == "__main__":
    main()
