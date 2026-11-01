/* Compatibility launcher for the portable evidence-first literature workspace.
   This file contains no provider claim, corpus text, or credential. */
(()=>{'use strict';
  const link=document.createElement('a');
  link.className='assistant-open';
  link.href='assistant.html';
  link.textContent='Ask MOSAIC';
  link.setAttribute('aria-label','Open the Ask MOSAIC literature workspace');
  const params=new URL(location.href).searchParams;
  const query=params.get('q'),paper=params.get('paper');
  if(query||paper){const target=new URL(link.href,location.href);if(query)target.searchParams.set('q',query);if(paper)target.searchParams.set('paper',paper);link.href=target.pathname+target.search;}
  document.body.append(link);
})();
