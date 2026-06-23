
import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile

# Load Model
model = joblib.load("salary_predictor.pkl")


def predict_income(
    age,
    workclass,
    fnlwgt,
    education,
    education_num,
    marital_status,
    occupation,
    relationship,
    race,
    sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country
):

    data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country
    }])

    prediction = model.predict(data)[0]

    try:
        probability = model.predict_proba(data)[0][1]
    except:
        probability = 0.5

    if prediction == 1:

        return f"""
        <div style="
            background: linear-gradient(135deg,#00b894,#00cec9);
            padding:25px;
            border-radius:15px;
            color:white;
            text-align:center;
        ">
            <h2>💰 High Income Predicted</h2>
            <h1>> $50,000</h1>
            <h3>Confidence: {probability:.2%}</h3>
        </div>
        """

    return f"""
    <div style="
        background: linear-gradient(135deg,#e17055,#d63031);
        padding:25px;
        border-radius:15px;
        color:white;
        text-align:center;
    ">
        <h2>📉 Lower Income Predicted</h2>
        <h1>≤ $50,000</h1>
        <h3>Confidence: {(1-probability):.2%}</h3>
    </div>
    """


custom_css = """

.gradio-container{
    max-width:1400px !important;
    margin:auto;
}

#title{
    text-align:center;
}

.predict-btn{
    height:55px;
    font-size:18px;
    font-weight:bold;
}
"""


with gr.Blocks(css=custom_css) as demo:

    gr.Markdown(
        """
        # 💰 AI Salary Prediction System

        @ PREDICTING PERSON'S ANNUAL INCOME.
        """,
        elem_id="title"
    )

    with gr.Row():

        with gr.Column(scale=2):

            age = gr.Number(value=35, label="Age")

            workclass = gr.Dropdown(
                [
                    'Private',
                    'Self-emp-not-inc',
                    'Local-gov',
                    'State-gov',
                    'Self-emp-inc',
                    'Federal-gov',
                    'Without-pay'
                ],
                value="Private",
                label="Workclass"
            )

            fnlwgt = gr.Number(
                value=189778,
                label="Final Weight"
            )

            education = gr.Dropdown(
                [
                    'HS-grad',
                    'Some-college',
                    'Bachelors',
                    'Masters',
                    'Assoc-voc',
                    'Doctorate'
                ],
                value="Bachelors",
                label="Education"
            )

            education_num = gr.Number(
                value=13,
                label="Education Number"
            )

            marital_status = gr.Dropdown(
                [
                    'Married-civ-spouse',
                    'Never-married',
                    'Divorced',
                    'Separated'
                ],
                value='Married-civ-spouse',
                label='Marital Status'
            )

            occupation = gr.Dropdown(
                [
                    'Exec-managerial',
                    'Prof-specialty',
                    'Sales',
                    'Tech-support',
                    'Craft-repair'
                ],
                value='Exec-managerial',
                label='Occupation'
            )

            relationship = gr.Dropdown(
                [
                    'Husband',
                    'Wife',
                    'Own-child',
                    'Not-in-family'
                ],
                value='Husband',
                label='Relationship'
            )

            race = gr.Dropdown(
                [
                    'White',
                    'Black',
                    'Asian-Pac-Islander',
                    'Other'
                ],
                value='White',
                label='Race'
            )

            sex = gr.Dropdown(
                ['Male', 'Female'],
                value='Male',
                label='Sex'
            )

            capital_gain = gr.Number(
                value=0,
                label='Capital Gain'
            )

            capital_loss = gr.Number(
                value=0,
                label='Capital Loss'
            )

            hours_per_week = gr.Number(
                value=40,
                label='Hours Per Week'
            )

            native_country = gr.Dropdown(
                [
                    'United-States',
                    'India',
                    'Canada',
                    'Germany',
                    'England'
                ],
                value='United-States',
                label='Native Country'
            )

            predict_btn = gr.Button(
                "🚀 Predict Salary",
                elem_classes="predict-btn"
            )

        with gr.Column(scale=1):

            gr.Markdown("### Prediction Dashboard")

            result = gr.HTML()

            gr.Markdown(
                """
                ### Model Information

                ✅ LGBM Classifier

                📊 Binary Classification

                🎯 Income Prediction

                ⚡ Real-time Inference
                """
            )

    predict_btn.click(
        fn=predict_income,
        inputs=[
            age,
            workclass,
            fnlwgt,
            education,
            education_num,
            marital_status,
            occupation,
            relationship,
            race,
            sex,
            capital_gain,
            capital_loss,
            hours_per_week,
            native_country
        ],
        outputs=result
    )

# Create gauge chart
def create_confidence_chart(probability):

    fig, ax = plt.subplots(figsize=(5, 3))

    labels = ["<=50K", ">50K"]
    values = [1 - probability, probability]

    ax.bar(labels, values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.savefig(temp_file.name, bbox_inches="tight")
    plt.close()

    return temp_file.name
       

if __name__ == "__main__":
    demo.launch()



