import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
from pygooglenews import GoogleNews
import io
import os
import openai

# --- Functions ---

# Function to get the SambaNova client securely
def get_sambanova_client():
    """
    Initializes and returns a SambaNova API client.
    
    It fetches the API key from the environment variable 'SAMBANOVA_API_KEY'.
    """
    sambanova_api_key = "be38bb58-e3a4-422f-8e08-bdbbc6bbdc43"
    if not sambanova_api_key:
        st.error("SAMBANOVA_API_KEY environment variable is not set. Please set it in your environment.")
        return None
        
    client = openai.OpenAI(
        base_url="https://api.sambanova.ai/v1",
        api_key=sambanova_api_key,
    )
    return client

# --- Prompts ---
PROMPT_INDIVIDUAL_ANALYSIS = """Carefully analyze the following news article text for information directly indicating potential reasons for client churn specifically for an **employee benefits company in India**. Focus only on details that would impact an employee benefits provider or suggest a company might reduce or discontinue its employee benefits programs.

**Text:**
{provided_text}

Based on your analysis and using the provided categories below, determine the churn risk and the specific reason(s).

1.  **Risk Level (First Line):** State the risk level as one of the following:
    * "High Risk"
    * "Medium Risk"
    * "Low Risk"
    * "No Churn Risk Indicated" (If no relevant information is found regarding churn for an employee benefits company)

2.  **Reason(s) for Risk (Second Line):** If a risk is indicated, explain the major reason(s) concisely, referencing the relevant category (e.g., "Reason: [Category Name] - Brief explanation."). If there are multiple relevant reasons, list them clearly.

3.  **2-Line Summary of Analysis (Third and Fourth Lines):** Provide a brief, overall summary of the article's relevance to churn for an employee benefits company, condensing the key findings into exactly two lines. If no churn risk is indicated, summarize why the article is not relevant.

**Categories for Reasons:**
I. Corporate Restructuring (Mergers, Acquisitions, Joint Ventures, IPO, Entity Realignment, Rebranding, Consolidation, Subsidiary changes)
II. Business Discontinuity (Closures, Market Exits, Bankruptcy, Operational Suspensions, Business Model Pivots)
III. Strategic Policy Changes (Benefits Strategy Transformation, Leadership Changes impacting strategy, Cost Optimization related to benefits, Changes in top leadership impacting benefits)
IV. Financial Constraints (Cash Flow Issues, Cost-Cutting impacting benefits, Budget Reallocation away from benefits, Severe financial loss)
V. Employment Structure Changes (Workforce Reorganization, Shifts to contractual work, Remote work transitions impacting benefits, Layoffs, Furloughs, Downsizing)
VI. Regulatory & Compliance Factors (India Specific: Changes in tax policy, GST, labor codes, social security impacting benefits compliance or costs)
VII. Competitive Market Dynamics (Client switched vendor, New platform adoption by client, Competitor activity in benefits space, Pricing pressures on benefits, Market share shifts impacting client's ability to offer benefits, Disruption in client's industry affecting benefits, Client's value proposition change impacting benefits)
VIII. Technological Transitions (Digital transformation affecting benefits administration, HRMS integration impacting benefits systems, API changes relevant to benefits platforms, Analytics adoption impacting benefits, Mobile app for benefits, Platform upgrade for benefits management)
IX. Service Delivery Issues (Onboarding delay with benefits provider, Tech issues with benefits platform, Merchant issue impacting benefits, Support problem with benefits services, Delivery delay of benefits, Reimbursement issue with benefits claims)
X. Employee Engagement (Low adoption of benefits programs, Poor user experience with benefits platform, Negative employee feedback on benefits, Generation gap affecting benefits appeal, Hybrid work models impacting benefits usage, Usage drop in benefits offerings)

**Example Output Format (for High/Medium/Low Risk):**
High Risk
Reason: Business Discontinuity - Company announced complete shutdown impacting all operations including benefits.
Summary: The company is facing imminent closure, directly impacting its ability to retain any employee benefits plans. This represents a critical churn event for any associated benefits provider.

**Example Output Format (for No Risk):
No Churn Risk Indicated
Summary: The article discusses general market trends not specific to the company's operational or financial health. It provides no indication of changes relevant to employee benefits or potential churn.
"""

PROMPT_COMBINED_ANALYSIS = """Given the individual analyses of news articles related to a company and potential client churn, provide an overall summary (at most 4 lines).

**Individual Article Analyses:**
{individual_analyses_summary}

In the first line, state the overall risk level for churn for the company (e.g., "Overall High Risk," "Overall Medium Risk," "Overall Low Risk," "Overall No Churn Risk Indicated"). In the subsequent lines, summarize the major reasons for this overall risk, drawing from the categories mentioned in the individual analyses. Be concise and focus on the most impactful reasons across all articles. If no relevant information is found across all articles, state "Overall No Churn Risk Indicated."
"""

# Cache results for 1 hour to avoid repeated API calls
@st.cache_data(ttl=3600)
def analyze_text(company_name, provided_text, prompt_template, _sambanova_client):
    """Analyzes the provided text for churn indicators using SambaNova AI."""
    prompt = prompt_template.format(
        company_name=company_name, provided_text=provided_text)
    try:
        response = _sambanova_client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        output = response.choices[0].message.content
        return output if output else f"Unexpected response: {output}"
    except Exception as e:
        st.error(f"Error querying SambaNova AI for {company_name}: {e}")
        return "Analysis failed due to AI service error."


@st.cache_data(ttl=3600)  # Cache news fetching for 1 hour
def fetch_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None):
    """
    Fetches news articles for a given company using the pygooglenews library.
    Filters articles by allowed domains.
    """
    gn = GoogleNews(lang='en', country='IN')
    results = []
    if queries is None:
        queries = [company_name]

    try:
        # Process queries in groups of 3 to optimize API calls
        for i in range(0, len(queries), 3):
            group_queries = queries[i:i+3]
            combined_query = " OR ".join(group_queries)
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            search_results = gn.search(
                combined_query, from_=from_date_str, to_=to_date_str)

            if search_results and 'entries' in search_results:
                articles_for_query = []
                if allowed_domains:
                    for article in search_results['entries']:
                        source_link = article.get('source', {}).get('href', '')
                        parsed_uri = urlparse(source_link)
                        domain = parsed_uri.netloc.replace('www.', '')
                        if any(d in domain for d in allowed_domains):
                            articles_for_query.append(article)
                    # If no articles from allowed domains, add the top article as a fallback
                    if not articles_for_query and search_results['entries']:
                        articles_for_query.append(search_results['entries'][0])
                else:
                    articles_for_query = search_results['entries']

                results.extend(articles_for_query[:max_articles])
            else:
                st.warning(
                    f"No results or 'entries' not found for query '{combined_query}'")
    except Exception as e:
        st.error(f"Error fetching news for {company_name}: {e}")
        return None
    # Ensure total articles returned is at most max_articles
    return results[:max_articles]


def process_article(article):
    """Extracts summary or title from a news article."""
    return article.get('summary') or article.get('title') or ""


def analyze_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None, sambanova_client=None):
    """
    Fetches news articles for a company and analyzes them for churn indicators.
    """
    if not sambanova_client:
        return {"individual_analyses": [], "overall_summary": "API client is not available."}
        
    st.subheader(f"Analyzing News for **{company_name}**")
    all_articles = fetch_news(company_name, from_date,
                             to_date, max_articles, queries, allowed_domains)

    if not all_articles:
        return {"individual_analyses": [], "overall_summary": "No relevant news articles found for analysis."}

    individual_analyses_list = []
    combined_analysis_text_for_model = ""

    for i, article in enumerate(all_articles):
        article_text = process_article(article)
        article_url = article.get('link', 'No URL available')
        # Get actual title or fallback
        article_title = article.get('title', f"Article {i+1}")

        if article_text:
            analysis_result = analyze_text(
                company_name, article_text, PROMPT_INDIVIDUAL_ANALYSIS, sambanova_client)
            individual_analyses_list.append({
                "title": article_title,  # Store the title
                "url": article_url,
                "analysis": analysis_result
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{analysis_result}\n\n"
        else:
            no_text_analysis = "No Churn Risk Indicated (No text in article summary/title)."
            individual_analyses_list.append({
                "title": article_title,  # Store the title even if no text
                "url": article_url,
                "analysis": no_text_analysis
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{no_text_analysis}\n\n"

    overall_summary_result = "Overall No Churn Risk Indicated."
    if individual_analyses_list:
        combined_prompt = PROMPT_COMBINED_ANALYSIS.format(
            individual_analyses_summary=combined_analysis_text_for_model.strip())
        overall_summary_result = analyze_text(
            company_name, combined_prompt, "{provided_text}", sambanova_client)

    return {"individual_analyses": individual_analyses_list, "overall_summary": overall_summary_result}


def get_risk_level(summary_text):
    """Extracts risk level from a summary string."""
    summary_text_lower = summary_text.lower()
    if "low risk" in summary_text_lower:
        return "Low Risk"
    elif "medium risk" in summary_text_lower:
        return "Medium Risk"
    elif "high risk" in summary_text_lower:
        return "High Risk"
    elif "no churn risk indicated" in summary_text_lower:
        return "No Churn Risk Indicated"
    return "Unknown Risk"  # Fallback


def display_summary_with_color(company_name, summary_text):
    """Displays the summary with color coding based on risk level."""
    risk_level = get_risk_level(summary_text)

    st.markdown(f"### Summary for {company_name}")  # New heading format

    if "High Risk" in risk_level:
        st.error(summary_text)
    elif "Medium Risk" in risk_level:
        st.warning(summary_text)
    elif "Low Risk" in risk_level:
        st.info(summary_text)
    else:
        # For "No Churn Risk Indicated" and "Unknown Risk"
        st.success(summary_text)


def run_analysis(company_names, days_to_search, custom_keyword_string=None):
    """Main function to orchestrate the news fetching and analysis for multiple companies."""
    results = {}
    today = datetime.today()
    # Use user-inputted days for the date range
    from_date = today - timedelta(days=days_to_search)
    max_articles_per_query = 10

    # Get the SambaNova client before starting the loop
    sambanova_client = get_sambanova_client()
    if not sambanova_client:
        return {}
        
    # --- DEFAULT CHURN KEYWORDS ---
    default_churn_keywords = {
        "Corporate Restructuring": [
            "merger", "acquisition", "investment", "joint venture", "IPO", "restructuring",
            "realignment", "rebranding", "subsidiary", "consolidation"
        ],
        "Business Discontinuity": [
            "shutdown", "closed", "bankruptcy", "insolvency", "pivot", "market exit"
        ],
        "Strategic Policy Changes": [
            "benefits withdrawn", "benefits discontinued", "centralization",
            "new CEO", "cost cutting", "budget cuts", "strategy shift"
        ],
        "Financial Constraints": [
            "payroll issue", "financial loss", "cost pressure", "cash flow", "budget reallocation"
        ],
        "Employment Structure Changes": [
            "employee transfer", "contractual workforce", "remote work",
            "layoffs", "furloughs", "downsizing"
        ],
        "Regulatory & Compliance": [
            "tax policy", "labor law", "income tax", "GST change", "budget amendment", "social security"
        ],
        "Competitive Market Dynamics": [
            "switched vendor", "new platform", "competitor", "pricing", "market share",
            "disruption", "value proposition"
        ],
        "Technological Transitions": [
            "digital transformation", "HRMS integration", "API", "analytics",
            "mobile app", "platform upgrade"
        ],
        "Service Delivery Issues": [
            "onboarding delay", "tech issues", "merchant issue", "support problem",
            "delivery delay", "reimbursement issue"
        ],
        "Employee Engagement": [
            "low adoption", "user experience", "employee feedback",
            "generation gap", "hybrid work", "usage drop"
        ]
    }

    # Process custom keywords from text area
    custom_keywords_flat_list = []
    if custom_keyword_string:
        # Split by comma and strip whitespace
        custom_keywords_flat_list = [
            kw.strip() for kw in custom_keyword_string.split(',') if kw.strip()]
        # For simplification, we'll put all custom keywords into a single 'Custom' category
        if custom_keywords_flat_list:
            churn_keywords_to_use = {"Custom": custom_keywords_flat_list}
        else:
            # Fallback if string is empty after stripping
            churn_keywords_to_use = default_churn_keywords
    else:
        churn_keywords_to_use = default_churn_keywords

    # --- YOUR SPECIFIED ALLOWED DOMAINS (UNCHANGED) ---
    allowed_domains = [
        "livemint.com", "economictimes.indiatimes.com", "business-standard.com",
        "thehindubusinessline.com", "financialexpress.com", "ndtvprofit.com",
        "zeebiz.com", "moneycontrol.com", "bloombergquint.com",
        "cnbctv18.com", "businesstoday.in", "indianexpress.com",
        "thehindu.com", "reuters.com", "businesstraveller.com",
        "sify.com", "telegraphindia.com", "outlookindia.com",
        "firstpost.com", "pulse.zerodha.com", "ndtvprofit.com",
        "ddnews.gov.in", "newsonair.gov.in", "pib.gov.in",
        "niti.gov.in", "rbi.org.in", "sebi.gov.in",
        "dpiit.gov.in", "investindia.gov.in", "indiabriefing.com",
        "Taxscan.in", "bwbusinessworld.com", "inc42.com",
        "yourstory.com", "vccircle.com", "entrackr.com",
        "the-ken.com", "linkedin.com", "mca.gov.in",
        "zaubacorp.com", "tofler.in", "smestreet.in"
    ]

    processed_allowed_domains = [domain.replace(
        "www.", "") for domain in allowed_domains]

    st.sidebar.subheader("Analysis Parameters")
    st.sidebar.info(
        f"Analyzing news from: **{from_date.strftime('%Y-%m-%d')}** to **{today.strftime('%Y-%m-%d')}** ({days_to_search} days)")
    st.sidebar.info(f"Max Articles per Query: **{max_articles_per_query}**")
    st.sidebar.info(
        f"Filtered by {len(processed_allowed_domains)} specified business news domains.")
    if custom_keyword_string:
        st.sidebar.info("Using **custom keywords** for search.")
    else:
        st.sidebar.info("Using **default keywords** for search.")

    for company in company_names:
        queries = [company] + [f"{company} {keyword}" for category_keywords in churn_keywords_to_use.values()
                               for keyword in category_keywords]
        company_analysis = analyze_news(
            company, from_date, today, max_articles_per_query, queries, processed_allowed_domains, sambanova_client
        )
        results[company] = company_analysis if company_analysis else {
            "overall_summary": "Analysis failed."}
    return results


# --- Streamlit App Layout ---
st.set_page_config(page_title="Company Churn Risk Analyzer", layout="wide")
st.title("💡 Company Churn Risk Analysis (India Focus)")
st.markdown("""
This application helps identify potential client churn risks for an employee benefits company in India by analyzing recent news articles.
Upload an **Excel** file containing company names, or enter them manually, and the app will fetch relevant news, summarize it, and provide churn risk assessments.
""")

# User input for number of days
days_to_search = st.slider(
    "Select the number of days for news search (back from today):",
    min_value=1,
    max_value=365,
    value=90,  # Default to 90 days
    step=1,
    help="This determines how far back in time the news articles will be fetched."
)

# File uploader widget for company names
uploaded_companies_file = st.file_uploader(
    "Upload your 'company_names.xlsx' file (Optional)", type=["xlsx"], key="companies_upload")

company_names_from_upload = []
if uploaded_companies_file is not None:
    try:
        company_df = pd.read_excel(uploaded_companies_file)
        if "CompanyName" in company_df.columns:
            company_names_from_upload = company_df["CompanyName"].dropna(
            ).tolist()
            if company_names_from_upload:
                st.success(
                    f"Successfully loaded **{len(company_names_from_upload)}** companies from **'{uploaded_companies_file.name}'**.")
            else:
                st.warning(
                    "The 'CompanyName' column is empty after loading. Please check your Excel file.")
        else:
            st.error(
                "Error: The uploaded Excel file must contain a **'CompanyName'** column.")
    except Exception as e:
        st.error(
            f"Error reading Excel file: {e}. Please ensure it's a valid Excel file with the correct column name.")


# Custom input for company names (optional fallback)
st.markdown("---")
st.subheader("Or enter company names manually (comma-separated):")
manual_company_input = st.text_input(
    "Example: Reliance Industries, Tata Consultancy Services, Wipro")

company_names_to_analyze = []
if manual_company_input:
    manual_companies = [c.strip()
                        for c in manual_company_input.split(',') if c.strip()]
    if manual_companies:
        st.info(
            f"Using manually entered companies: **{', '.join(manual_companies)}**")
        company_names_to_analyze = manual_companies
else:
    company_names_to_analyze = company_names_from_upload


# --- Keyword Customization (Text Area) ---
st.markdown("---")
st.subheader("Custom Search Keywords (Optional)")
st.info("Enter your custom keywords below, separated by commas. If this field is left empty, the default keywords will be used.")

custom_keywords_input = st.text_area(
    "Enter keywords (e.g., layoff, acquisition, new CEO, financial trouble)",
    height=100,
    help="Each keyword will be searched alongside the company name. Separate multiple keywords with commas."
)


# Display default keywords
st.markdown("---")
st.subheader("Default Churn Keywords for Reference")
with st.expander("Click to view default keywords"):
    default_churn_keywords_display = {
        "Corporate Restructuring": [
            "merger", "acquisition", "investment", "joint venture", "IPO", "restructuring",
            "realignment", "rebranding", "subsidiary", "consolidation"
        ],
        "Business Discontinuity": [
            "shutdown", "closed", "bankruptcy", "insolvency", "pivot", "market exit"
        ],
        "Strategic Policy Changes": [
            "benefits withdrawn", "benefits discontinued", "centralization",
            "new CEO", "cost cutting", "budget cuts", "strategy shift"
        ],
        "Financial Constraints": [
            "payroll issue", "financial loss", "cost pressure", "cash flow", "budget reallocation"
        ],
        "Employment Structure Changes": [
            "employee transfer", "contractual workforce", "remote work",
            "layoffs", "furloughs", "downsizing"
        ],
        "Regulatory & Compliance": [
            "tax policy", "labor law", "income tax", "GST change", "budget amendment", "social security"
        ],
        "Competitive Market Dynamics": [
            "switched vendor", "new platform", "competitor", "pricing", "market share",
            "disruption", "value proposition"
        ],
        "Technological Transitions": [
            "digital transformation", "HRMS integration", "API", "analytics",
            "mobile app", "platform upgrade"
        ],
        "Service Delivery Issues": [
            "onboarding delay", "tech issues", "merchant issue", "support problem",
            "delivery delay", "reimbursement issue"
        ],
        "Employee Engagement": [
            "low adoption", "user experience", "employee feedback",
            "generation gap", "hybrid work", "usage drop"
        ]
    }
    for category, keywords in default_churn_keywords_display.items():
        st.markdown(f"**{category}**: {', '.join(keywords)}")


if st.button("🚀 Start Analysis"):
    if not company_names_to_analyze:
        st.warning(
            "No company names available for analysis. Please upload an Excel file or enter names manually.")
    else:
        with st.spinner("Crunching numbers and fetching news... This might take a while for each company."):
            analysis_results = run_analysis(
                company_names_to_analyze, days_to_search, custom_keywords_input)  # Pass custom keywords input

        st.success("🎉 Analysis Complete!")
        st.markdown("---")

        # Display the results
        for company, analysis in analysis_results.items():
            st.markdown(f"## :office: {company}")

            # Display Overall Churn Risk Summary with color and new heading
            display_summary_with_color(company, analysis.get(
                "overall_summary", "No overall analysis available."))

            st.markdown("### Individual Article Analyses")
            if analysis.get("individual_analyses"):
                for i, article_analysis in enumerate(analysis["individual_analyses"]):
                    # Display actual article title
                    st.markdown(
                        f"#### :newspaper: {article_analysis['title']}")
                    st.markdown(f"**URL:** [Link]({article_analysis['url']})")
                    article_analysis_text = article_analysis['analysis']
                    # Color-code individual analysis
                    if "High Risk" in article_analysis_text:
                        st.error(f"**Analysis:** {article_analysis_text}")
                    elif "Medium Risk" in article_analysis_text:
                        st.warning(f"**Analysis:** {article_analysis_text}")
                    elif "Low Risk" in article_analysis_text:
                        st.info(f"**Analysis:** {article_analysis_text}")
                    else:
                        st.write(f"**Analysis:** {article_analysis_text}")
                    st.markdown("---")
            else:
                st.info("No individual articles found for detailed analysis.")
            st.markdown("---")  # Separator between companies

        # Export to Excel
        data_for_df = []
        for company, analysis in analysis_results.items():
            overall_summary = analysis.get(
                "overall_summary", "No analysis available")
            overall_risk_level = get_risk_level(
                overall_summary)  # Get overall risk level

            company_data = {
                "Company": company,
                "Overall Risk Level": overall_risk_level,  # New column for risk level
                "Overall Summary": overall_summary
            }
            for i, article_analysis in enumerate(analysis.get("individual_analyses", [])):
                # Get individual article risk level
                article_risk_level = get_risk_level(
                    article_analysis["analysis"])
                company_data[f"Article {i+1} Title"] = article_analysis["title"]
                company_data[f"Article {i+1} URL"] = article_analysis["url"]
                # Add risk level for individual article
                company_data[f"Article {i+1} Risk Level"] = article_risk_level
                company_data[f"Article {i+1} Analysis"] = article_analysis["analysis"]
            data_for_df.append(company_data)

        if data_for_df:
            df_results = pd.DataFrame(data_for_df)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file_name = f"churn_analysis_results_{timestamp}.xlsx"

            excel_buffer = io.BytesIO()
            df_results.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_buffer.seek(0)

            st.download_button(
                label="Download All Results as Excel 📊",
                data=excel_buffer,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Click to download the comprehensive analysis results."
            )
            st.success("Results are ready for download!")
        else:
            st.warning(
                "No data to export to Excel, as no analysis was performed.")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app leverages the **OpenRouter** API and specifically the free `meta-llama/llama-3.3-70b-instruct` model, along with Google News, for churn risk analysis.")
