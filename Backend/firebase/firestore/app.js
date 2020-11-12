const TODOList = document.querySelector('#TODO-list');
const form = document.querySelector('#add-todo-form');

function render(doc) {
    let li = document.createElement('li');
    let title = document.createElement('span');
    let time = document.createElement('span');
    let cross = document.createElement('div');
    cross.textContent = 'x';
    cross.addEventListener('click', (e) => {
        e.stopPropagation();
        let id = e.target.parentElement.getAttribute('data-id');
        db.collection('TODO').doc(id).delete();
    })

    li.setAttribute('data-id', doc.id);
    title.textContent = doc.data().title;
    time.textContent = new Date(doc.data().time.seconds * 1000).toLocaleDateString("zh-Hans-CN");

    li.appendChild(title);
    li.appendChild(time);
    li.appendChild(cross);

    TODOList.appendChild(li);
}

db.collection('TODO').orderBy('time').onSnapshot(snapshot => {
    let changes = snapshot.docChanges();
    changes.forEach(change => {
        console.log(change);
        if (change.type == 'added') {
            render(change.doc);
        }
        else if (change.type == 'removed') {
            let li = TODOList.querySelector('[data-id=' + change.doc.id + ']');
            TODOList.removeChild(li);
        }
    });
})

form.addEventListener('submit', (e) => {
    e.preventDefault();
    db.collection('TODO').add({
        title: form.title.value,
        time: {
            seconds: Date.parse(form.time.value) / 1000,
            nanoseconds: 0
        }
    });
    form.title.value = '';
    form.time.value = '';
})