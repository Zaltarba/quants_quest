--- 
layout: page 
title:
permalink: /teaching/ 
---

<style>
.teaching-container {
    max-width: 900px;
    margin: auto;
}

.course-title {
    text-align: center;
    margin-bottom: 30px;
}

.course-title h1 {
    margin-bottom: 10px;
}

.course-intro {
    text-align: center;
    font-size: 1.05em;
    color: #444;
}

    .sessions-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 30px;
}

.session-card {
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 14px 16px;
    background: #fafafa;
    text-align: center;
}

.session-card h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.05em;
    color: #2c3e50;
}

.session-links {
    display: flex;
    justify-content: center;
    gap: 20px;
}

.session-links a {
    color: #3498db;
    font-weight: 500;
    text-decoration: none;
}

.session-links a:hover {
    text-decoration: underline;
}

/* Mobile: 1 session per row */
@media (max-width: 768px) {
    .sessions-grid {
        grid-template-columns: 1fr;
    }
}

    
.assessment-box {
    border-left: 5px solid #3498db;
    background: #f4f8fc;
    padding: 20px;
    border-radius: 6px;
    margin-top: 30px;
}

.resources {
    margin-top: 30px;
}

.resources a {
    color: #3498db;
    font-weight: 500;
    text-decoration: none;
}
</style>

<div class="teaching-container">

<div class="course-title">
    <h1>Dauphine PSL – Master 222</h1>
    <p class="course-intro">
        Introduction to Python (2025) 
    </p>
</div>

<p class="course-intro">
    All course materials and project instructions are available on this
    <a href="https://github.com/Zaltarba/PSL_python_for_finance/tree/main">GitHub repository</a>.
</p>

<div class="sessions-grid">

    <div class="session-card">
        <h3>📘 Session 1</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_1.ipynb">Exercises</a>
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_1_corrected.ipynb">Solutions</a>
        </div>
    </div>

    <div class="session-card">
        <h3>📘 Session 2</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_2.ipynb">Exercises</a>
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_2_corrected.ipynb">Solutions</a>
        </div>
    </div>

    <div class="session-card">
        <h3>📘 Session 3</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_3.ipynb">Exercises</a>
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_3_corrected.ipynb">Solutions</a>
        </div>
    </div>

    <div class="session-card">
        <h3>📘 Session 4</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_4.ipynb">Exercises</a>
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_4_corrected.ipynb">Solutions</a>
        </div>
    </div>

    <div class="session-card">
        <h3>📘 Session 5</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_5.ipynb">Exercises</a>
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_5_corrected.ipynb">Solutions</a>
        </div>
    </div>

    <div class="session-card">
        <h3>📘 Session 6</h3>
        <div class="session-links">
            <a href="https://colab.research.google.com/github/Zaltarba/PSL_python_for_finance/blob/main/python_session_6.ipynb">Exercises</a>
        </div>
    </div>

</div>

<div class="assessment-box">
    <h3>📝 Assessments</h3>

    <p><strong>Paper Exam</strong><br>
    6 November 2025<br>
    50% of final grade</p>

    <p><strong>Group Project</strong><br>
    Due 9 January 2025 (oral presentation)<br>
    50% of final grade</p>
</div>

<div class="resources">
    <h3>📚 Resources</h3>
    <a href="https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf">Pandas Cheat Sheet</a>
</div>

</div>
