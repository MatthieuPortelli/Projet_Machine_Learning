from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer
from datetime import datetime
import io


def convert_fig_to_image(fig):
    # Convertir la figure en image
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    return img_buffer


def generate_pdf_report(selected_model, best_params, best_score, metrics, fig_1, fig_2):
    # Créez un objet StringIO pour stocker le PDF généré
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    # Définissez un style pour le contenu du rapport
    styles = getSampleStyleSheet()
    # Créez le contenu du rapport en utilisant Platypus
    content = []

    # Page de titre
    title = "Rapport d'Analyse Machine Learning"
    today_date = datetime.today().strftime('%d-%m-%Y')
    content.append(Paragraph(title, styles['Title']))
    content.append(Paragraph(today_date, styles['Title']))
    content.append(Spacer(1, 12))

    # Section 1 : Informations sur le modèle
    section_title = "Informations sur le Modèle"
    content.append(Paragraph(section_title, styles['Heading1']))
    content.append(Spacer(1, 12))
    # Modèle sélectionné
    content.append(Paragraph(f"<b>Modèle sélectionné :</b> {selected_model}", styles['Normal']))
    content.append(Spacer(1, 12))
    # Meilleurs paramètres
    content.append(Paragraph(f"<b>Meilleurs paramètres :</b>", styles['Normal']))
    for param_name, param_value in best_params.items():
        content.append(Paragraph(f"- {param_name} : {param_value}", styles['Normal']))
    content.append(Spacer(1, 12))
    # Meilleur score
    content.append(Paragraph(f"<b>Meilleur score :</b> {best_score}", styles['Normal']))
    content.append(Spacer(1, 12))
    # Métriques
    content.append(Paragraph(f"<b>Métriques :</b>", styles['Normal']))
    for metric_name, metric_value in metrics.items():
        content.append(Paragraph(f"- {metric_name} : {metric_value}", styles['Normal']))
    content.append(Spacer(1, 12))

    # Section 2 : Visualisations
    section_title = "Visualisations"
    content.append(Paragraph(section_title, styles['Heading1']))
    content.append(Spacer(1, 12))
    # Visualisations 1
    content.append(Paragraph("<b>Visualisation 1 :</b>", styles['Normal']))
    img1 = Image(convert_fig_to_image(fig_1), width=400, height=300)
    content.append(img1)
    content.append(Spacer(1, 12))
    # Visualisations 2
    content.append(Paragraph("<b>Visualisation 2 :</b>", styles['Normal']))
    img2 = Image(convert_fig_to_image(fig_2), width=400, height=300)
    content.append(img2)
    content.append(Spacer(1, 12))

    # Ajoutez le contenu au PDF
    doc.build(content)
    # Obtenez le contenu du PDF à partir du tampon
    pdf_data = pdf_buffer.getvalue()
    # Fermez le tampon
    pdf_buffer.close()
    return pdf_data
