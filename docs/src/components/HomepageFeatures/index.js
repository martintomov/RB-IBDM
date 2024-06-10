import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'What is InsectSAM?',
    Img: require('@site/static/img/undraw_docusaurus_mountain.png').default,
    description: (
      <>
      <b>InsectSAM</b> is an open-source machine learning model tailored for the <b>DIOPSIS camera systems
      and ARISE algorithms</b>, dedicated to Insect Biodiversity Detection and Monitoring in the Netherlands. 
      </>
    ),
  },
  {
    title: 'How does it work?',
    Img: require('@site/static/img/undraw_docusaurus_react.png').default,
    description: (
      <>
      Based on <b>Meta AI segment-anything</b>, InsectSAM is fine-tuned to accurately segment insects from complex backgrounds. It boosts the precision and efficiency of biodiversity monitoring algorithms, especially in scenarios with diverse backgrounds that attract insects.
      </>
    ),
  },
  {
    title: 'Technologies Used',
    Img: require('@site/static/img/undraw_docusaurus_stack.png').default,
    description: (
      <>
      <b>Python, PyTorch, Hugging Face Transformers, OpenCV,</b> and more. InsectSAM is designed to be easily integrated into existing DIOPSIS and ARISE algorithms, providing a seamless experience for researchers and developers.
      </>
    ),
  },
];

function Feature({Img, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={Img} className={styles.featureImg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
